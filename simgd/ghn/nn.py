from src.ghn.encoder import MLPEncoder
import torch
import torch.nn as nn
import os
#from .mlp import MLP
from .gnn.gatedgnn import GatedGNN
from .decoder import MLPDecoder#, ConvDecoder
#from .layers import ShapeEncoder
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
                 num_classes,
                 hypernet='gatedgnn',
                 decoder='conv',
                 weight_norm=False,
                 ve=False,
                 layernorm=False,
                 hid=32,
                 debug_level=0):
        super(GHN, self).__init__()

        assert len(max_shape) == 4, max_shape
        self.layernorm = layernorm
        self.weight_norm = weight_norm
        self.ve = ve
        self.debug_level = debug_level
        self.num_classes = num_classes
        self.hid = hid

        if layernorm:
            self.ln = nn.LayerNorm(hid)

        if hypernet == 'gatedgnn':
            self.gnn = GatedGNN(in_features=hid, ve=ve)
        else:
            raise NotImplementedError(hypernet)
        #elif hypernet == 'mlp':
        #    self.gnn = MLP(in_features=hid, hid=(hid, hid))
        

        self.max_shape = max_shape # [64,64,3,3]

        self.conv_enc = MLPEncoder(self.max_shape,out_features=hid)
        self.conv_dec = MLPDecoder(self.max_shape,in_features=hid)

        self.linear_enc = MLPEncoder(max_shape[:2],out_features=hid)
        self.linear_dec = MLPDecoder(max_shape[:2],in_features=hid)

        self.bias_enc = nn.Linear(max_shape[0],hid)
        self.bias_dec = nn.Linear(hid,max_shape[0])

    def forward(self, net, graph):
        features = torch.zeros((graph.n_nodes,self.hid))
        for ind, param in enumerate(graph.node_params[1:]):
            weight = net.state_dict()[param]
            if weight.ndimension() == 4:
                in_weight = torch.zeros(self.max_shape)
                in_weight[:weight.size(0),:weight.size(1),:weight.size(2),:weight.size(3)] = weight
                features[ind+1,:] = self.conv_enc(in_weight.flatten())
            elif weight.ndimension() == 2:
                in_weight = torch.zeros(self.max_shape[:2])
                in_weight[:weight.size(0),:weight.size(1)] = weight
                features[ind+1,:] = self.linear_enc(in_weight.flatten())
            elif weight.ndimension() == 1:
                in_weight = torch.zeros(self.max_shape[0])
                in_weight[:weight.size(0)] = weight
                features[ind+1,:] = self.bias_enc(in_weight)

        x = self.gnn(features, graph.edges, graph.node_feat[:,1])

        if self.layernorm:
            x = self.ln(x)

        out = []
        for ind, param in enumerate(graph.node_params[1:]):
            weight = net.state_dict()[param]
            if weight.ndimension() == 4:
                out.append(self.conv_dec(x[ind+1,:])[:weight.size(0),:weight.size(1),:weight.size(2),:weight.size(3)])
            elif weight.ndimension() == 2:
                out.append(self.linear_dec(x[ind+1,:])[:weight.size(0),:weight.size(1)])
            elif weight.ndimension() == 1:
                out.append(self.bias_dec(x[ind+1,:])[:weight.size(0)])

        return out
