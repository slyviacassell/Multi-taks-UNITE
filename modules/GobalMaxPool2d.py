# coding=utf-8

'''
Created: 2021/3/1
@author: Slyviacassell@github.com
'''

import torch
import torch.nn as nn


class GlobalMaxPool2d(nn.Module):
    def __init__(self, return_indices=False):
        super(GlobalMaxPool2d, self).__init__()
        self.return_indices = return_indices

    def forward(self, x: torch.Tensor):
        x_size = x.size()
        out = x.view(*x_size[:-2], -1)
        out, indices = out.max(dim=-1)
        out = out.view(*x_size[:-3], -1).contiguous()
        if self.return_indices:
            return out, indices
        else:
            return out
