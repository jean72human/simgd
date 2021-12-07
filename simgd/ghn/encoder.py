import numpy as np
import torch.nn as nn
from .layers import get_activation, MLP


class MLPEncoder(nn.Module):
    def __init__(self,
                 in_shape,
                 hid=(64,),
                 out_features=32):
        super(MLPEncoder, self).__init__()

        assert len(hid) > 0, hid
        self.out_features = out_features
        self.mlp = MLP(in_features=np.prod(in_shape),
                       hid=(*hid, out_features),
                       activation='relu',
                       last_activation=None)


    def forward(self, x, max_shape=(0,0)):
        x = self.mlp(x)
        #if sum(max_shape) > 0:
        #    x = x[:, :, :, :max_shape[0], :max_shape[1]]
        return x