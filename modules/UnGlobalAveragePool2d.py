# coding=utf-8

'''
Created: 2021/3/1
@author: Slyviacassell@github.com
'''

import torch
import torch.nn as nn


class UnGlobalAveragePool2d(nn.Module):
    def __init__(self):
        super(UnGlobalAveragePool2d, self).__init__()

    def forward(self, x: torch.Tensor, size: tuple):
        assert len(x.size()) >= 2
        out = x.unsqueeze(-1).unsqueeze(-1)
        repeat_times = [1] * len(out.size()[:-2]) + [size[0], size[1]]
        out = out.repeat(*repeat_times)
        return out
