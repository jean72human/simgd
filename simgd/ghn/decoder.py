import numpy as np
import torch.nn as nn
from .layers import get_activation, MLP


class MLPDecoder(nn.Module):
    def __init__(self,
                 out_shape,
                 in_features=32,
                 hid=(64,)):
        super(MLPDecoder, self).__init__()

        assert len(hid) > 0, hid
        self.out_shape = out_shape
        self.mlp = MLP(in_features=in_features,
                       hid=(*hid, np.prod(out_shape)),
                       activation='relu',
                       last_activation='sigmoid')


    def forward(self, x, max_shape=(0,0)):
        x = self.mlp(x).view(*self.out_shape)
        #if sum(max_shape) > 0:
        #    x = x[:, :, :, :max_shape[0], :max_shape[1]]
        return x