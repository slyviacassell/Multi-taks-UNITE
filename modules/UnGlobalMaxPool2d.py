# coding=utf-8

'''
Created: 2021/3/1
@author: Slyviacassell@github.com
'''

import torch
import torch.nn as nn


class UnGlobalMaxPool2d(nn.Module):
    def __init__(self, pad=0):
        super(UnGlobalMaxPool2d, self).__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor, indices: torch.Tensor, size: tuple):
        assert len(x.size()) >= 2
        out = torch.zeros((*x.size(), *size))
        out_size = out.size()
        out = out.view(-1)
        out[indices.view(-1).add(
            torch.arange(0, len(indices.view(-1)), dtype=torch.int) * self.size[0] * self.size[1])] = x.view(-1)
        out = out.view(*out_size)
        return out
