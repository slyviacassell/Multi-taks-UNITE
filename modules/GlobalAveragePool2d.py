# coding=utf-8

'''
Created: 2021/3/1
@author: Slyviacassell@github.com
'''

import torch
import torch.nn as nn


class GlobalAveragePool2d(nn.Module):
    def __init__(self):
        super(GlobalAveragePool2d, self).__init__()

    def forward(self, x: torch.Tensor):
        assert len(x.size()) >= 2
        x_size = x.size()
        out = x.view(*x_size[:-2], -1)
        out = out.mean(dim=-1)
        out = out.view(*x_size[:-3], -1).contiguous()
        return out
