from typing import Mapping
import torch
import torch.nn as nn
import os
#from .mlp import MLP
from .gnn.gatedgnn import GatedGNN
from .data_store import DataStore
from .gnn.gcn import GCNModel
from .decoder import MLPDecoder, ConvDecoder
from .encoder import MLPEncoder, ConvEncoder
from .layers import ShapeEncoder
from ..deepnets1m.ops import NormLayers, PosEnc
from ..deepnets1m.genotypes import PRIMITIVES_DEEPNETS1M
from ..deepnets1m.net import named_layered_modules
from ..deepnets1m.graph import Graph, GraphBatch
from ..utils import capacity, default_device
import time



class GHN(nn.Module):
    r"""
    Graph HyperNetwork based on "Chris Zhang, Mengye Ren, Raquel Urtasun. Graph HyperNetworks for Neural Architecture Search. ICLR 2019."
    (https://arxiv.org/abs/1810.05749)
    """
    def __init__(self,
                 max_shape,
                 num_classes=10 ,
                 hypernet='gatedgnn',
                 decoder='conv',
                 weight_norm=False,
                 ve=False,
                 layernorm=False,
                 hid=32,
                 debug_level=0,
                 passes=1,
                 device="cpu",
                 weightnorm=False,
                 backmul=False):
        super(GHN, self).__init__()

        assert len(max_shape) == 4, max_shape
        self.layernorm = layernorm
        self.weight_norm = weight_norm
        self.ve = ve
        self.debug_level = debug_level
        self.num_classes = num_classes
        self.hid = hid
        self.weightnorm= weightnorm
        self.device = device

        if layernorm:
            self.ln = nn.LayerNorm(hid*3)

        if hypernet == 'gatedgnn':
            self.gnn = GatedGNN(in_features=hid*3, ve=ve, backmul=backmul, T=passes)
        elif hypernet == 'gcn':
            self.gnn = GCNModel(sz_in=hid*3,sz_out=hid*3)
        else:
            raise NotImplementedError(hypernet)

        self.hypernet = hypernet
        #elif hypernet == 'mlp':
        #    self.gnn = MLP(in_features=hid, hid=(hid, hid))

        self.max_shape = max_shape # [64,64,3,3]

        self.conv_enc = ConvEncoder(self.max_shape,out_features=hid)
        self.conv_dec = ConvDecoder(self.max_shape,in_features=hid*3)

        self.linear_enc = MLPEncoder(max_shape[:2],out_features=hid)
        self.linear_dec = MLPDecoder(max_shape[:2],in_features=hid*3)

        self.bias_enc = nn.Linear(max_shape[0],hid)
        self.bias_dec = nn.Linear(hid*3,max_shape[0])

        #self.predictor = nn.Linear(hid*4,1)

        self.layer_embed = nn.Embedding(len(PRIMITIVES_DEEPNETS1M)+1,hid)

    def forward(self, data, grads, graph):
        features = torch.zeros((graph.n_nodes,self.hid*3), device=self.device)
        features[0,:] = 1
        for ind, (name,param) in enumerate(graph.node_params[1:]):
            if param in data.keys():
                weight = data[param]
                grad = grads[param]
                if weight.ndimension() == 4:
                    in_grad = torch.zeros(self.max_shape,device=self.device)
                    in_grad[:weight.size(0),:weight.size(1),:weight.size(2),:weight.size(3)] = grad
                    in_weight = torch.zeros(self.max_shape,device=self.device)
                    in_weight[:weight.size(0),:weight.size(1),:weight.size(2),:weight.size(3)] = weight
                    features[ind+1,:self.hid] = self.conv_enc(in_weight)
                    features[ind+1,self.hid:self.hid*2] = self.conv_enc(in_weight)
                elif weight.ndimension() == 2:
                    in_grad = torch.zeros(self.max_shape[:2],device=self.device)
                    in_grad[:weight.size(0),:weight.size(1)] = grad
                    in_weight = torch.zeros(self.max_shape[:2],device=self.device)
                    in_weight[:weight.size(0),:weight.size(1)] = weight
                    features[ind+1,:self.hid] = self.linear_enc(in_weight)
                    features[ind+1,self.hid:self.hid*2] = self.linear_enc(in_weight)
                elif weight.ndimension() == 1:
                    in_grad = torch.zeros(self.max_shape[0],device=self.device)
                    in_grad[:weight.size(0)] = grad
                    in_weight = torch.zeros(self.max_shape[0],device=self.device)
                    in_weight[:weight.size(0)] = weight
                    features[ind+1,:self.hid] = self.bias_enc(in_weight)
                    features[ind+1,self.hid:self.hid*2] = self.bias_enc(in_weight)
            prim_ind = PRIMITIVES_DEEPNETS1M.index(name) if name in PRIMITIVES_DEEPNETS1M else len(PRIMITIVES_DEEPNETS1M)
            features[ind+1,self.hid*2:] = self.layer_embed(torch.tensor([prim_ind], device=self.device)).squeeze(0)

        if self.hypernet == 'gcn':
            x = self.gnn(features, (graph._Adj==1).long())
        else:
            x = self.gnn(features, graph.edges, graph.node_feat[:,1])

        if self.layernorm:
            x = self.ln(x)

        out = {}
        for ind, (name,param) in enumerate(graph.node_params[1:]):
            if param in data.keys():
                weight_dim = data[param].shape
                if len(weight_dim) == 4:
                    out[param] = self.conv_dec(x[ind+1,:])[:weight_dim[0],:weight_dim[1],:weight_dim[2],:weight_dim[3]]
                elif len(weight_dim) == 2:
                    out[param] = self.linear_dec(x[ind+1,:])[:weight_dim[0],:weight_dim[1]]
                elif len(weight_dim) == 1:
                    out[param] = self.bias_dec(x[ind+1,:])[:weight_dim[0]]
                    
                if self.weight_norm:
                    out[param] = self._normalize(out[param], len(weight_dim)>1)

                #out[param] = torch.tanh(out[param])
            
        return DataStore(out) 


    def _normalize(self, p, is_w):
        r"""
        Normalizes the predicted parameter tensor according to the Fan-In scheme described in the paper.
        :param module: nn.Module
        :param p: predicted tensor
        :param is_w: True if it is a weight, False if it is a bias
        :return: normalized predicted tensor
        """
        if p.dim() > 1:

            sz = p.shape

            no_relu = len(sz) > 2 and (sz[1] == 1 or sz[2] < sz[3])
            if no_relu:
                # layers not followed by relu
                beta = 1.
            else:
                # for layers followed by rely increase the weight scale
                beta = 2.

            # fan-out:
            # p = p * (beta / (sz[0] * p[0, 0].numel())) ** 0.5

            # fan-in:
            p = p * (beta / p[0].numel()) ** 0.5

        else:

            if is_w:
                p = 2 * torch.sigmoid(0.5 * p)  # BN/LN norm weight is [0,2]
            else:
                p = torch.tanh(0.2 * p)         # bias is [-1,1]

        return p

