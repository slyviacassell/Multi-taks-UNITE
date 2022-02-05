# coding=utf-8

'''
Created: 2021/3/12
@author: Slyviacassell@github.com
'''

import torch
import torch.nn as nn

from memonger import SublinearSequential


class Stage(nn.Module):
    def __init__(self, out_channels, layers):
        super(Stage, self).__init__()
        if isinstance(layers, (nn.Sequential, SublinearSequential)):
            self.feature = layers
        else:
            self.feature = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature(x)
        # for n, m in self.named_modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         print(m.running_mean, m.running_var)
        return out
