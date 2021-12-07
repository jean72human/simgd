
import torch
import torch.nn as nn
import numpy as np
import copy


def get_activation(activation):
    if activation is not None:
        if activation == 'relu':
            f = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            f = nn.LeakyReLU()
        elif activation == 'selu':
            f = nn.SELU()
        elif activation == 'elu':
            f = nn.ELU()
        elif activation == 'rrelu':
            f = nn.RReLU()
        elif activation == 'sigmoid':
            f = nn.Sigmoid()
        else:
            raise NotImplementedError(activation)
    else:
        f = nn.Identity()

    return f

class MLP(nn.Module):
    def __init__(self,
                 in_features=32,
                 hid=(32, 32),
                 activation='relu',
                 last_activation='same'):
        super(MLP, self).__init__()

        assert len(hid) > 0, hid
        fc = []
        for j, n in enumerate(hid):
            fc.extend([nn.Linear(in_features if j == 0 else hid[j - 1], n),
                       get_activation(last_activation if
                                      (j == len(hid) - 1 and
                                       last_activation != 'same')
                                      else activation)])
        self.fc = nn.Sequential(*fc)


    def forward(self, x, *args, **kwargs):
        if isinstance(x, tuple):
            x = x[0]
        return self.fc(x)