# coding=utf-8

'''
Created: 2021/3/1
@author: Slyviacassell@github.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from typing import (
    Dict
)

from .UNITELayer import UNITELayer
from .DeepLabV3Plus import DeepLabHeadV3Plus


class UNITENet(nn.Module):
    def __init__(self, backbones: Dict,
                 n_tasks,
                 n_patch,
                 reconstruct_src,
                 init_weights=[0.9, 0.1],
                 init_method='constant',
                 activation='relu',
                 batch_norm=True,
                 shortcut=False,
                 bn_before_activation=True,
                 conv_bias=False,
                 similarity='bilinear'):
        super(UNITENet, self).__init__()
        assert len(backbones) == n_tasks
        self.n_tasks = n_tasks
        self.backbones = nn.ModuleDict(backbones)

        # assume that all backbones have same num of stages and same architectures
        for k, b in backbones.items():
            assert len(n_patch) == len(b.backbone.stages)

        self.num_stages = len(n_patch)
        self.n_patch = n_patch
        unites = []
        total_channels = 0

        net1 = backbones['task_0']
        if isinstance(net1, DeepLabHeadV3Plus):
            self.is_low_level = True
        else:
            self.is_low_level = False
        for stage_id in range(self.num_stages):
            out_channels = net1.backbone.stages[stage_id].out_channels
            total_channels += out_channels
            _unite = UNITELayer(out_channels=out_channels,
                                n_patches=n_patch[stage_id],
                                reconstruct_src=reconstruct_src,
                                init_weights=init_weights,
                                init_method=init_method,
                                activation=activation,
                                batch_norm=batch_norm,
                                bn_before_activation=bn_before_activation,
                                conv_bias=conv_bias,
                                similarity=similarity,
                                n_tasks=n_tasks)
            unites.append(_unite)

        unites = nn.ModuleList(unites)

        # all tasks use same shortcut ?
        self.shortcut = shortcut
        final_conv = None
        if shortcut:
            print("Using multi-scale")
            conv = nn.Conv2d(total_channels, net1.backbone.stages[-1].out_channels, kernel_size=1, bias=conv_bias)
            bn = nn.BatchNorm2d(net1.backbone.stages[-1].out_channels, eps=1e-03, momentum=0.05)
            if bn_before_activation:
                print("Using bn before activation")
                final_conv = [conv, bn, nn.ReLU()]
            else:
                final_conv = [conv, nn.ReLU(), bn]
            final_conv = nn.Sequential(*final_conv)

        self.unites = nn.ModuleDict({
            'unites': unites,
            'shortcut': final_conv,
        })

    def forward(self, features: Dict) -> Dict:
        # assume all features have same size
        _, _, H, W = features['task_0'].size()

        task_shortcuts = []
        out = {**features}
        cls_fea = {'task_{}'.format(i): {} for i in range(self.n_tasks)}
        for stage_id in range(self.num_stages):
            out.update({k: self.backbones[k].backbone.stages[stage_id](v) for k, v in out.items()})
            out = self.unites['unites'][stage_id](out)

            if stage_id == 1 and self.is_low_level:
                for i in range(self.n_tasks):
                    cls_fea['task_{}'.format(i)].update({'low_level': out['task_{}'.format(i)]})

            if self.shortcut:
                task_shortcuts.append({k: v for k, v in out.items()})
        if self.shortcut:
            _, _, h, w = task_shortcuts[-1]['task_0'].size()
            for i in range(self.n_tasks):
                _out = torch.cat(
                    [F.interpolate(_x['task_{}'.format(i)], (h, w), mode='bilinear', align_corners=True) for _x in
                     task_shortcuts[:-1]] + [task_shortcuts[-1]['task_{}'.format(i)]], dim=1)
                _out = self.unites['shortcut'](_out)
                out.update({'task_{}'.format(i): _out})

        for i in range(self.n_tasks):
            cls_fea['task_{}'.format(i)].update({'out': out['task_{}'.format(i)]})
            _out = self.backbones['task_{}'.format(i)].classifier(cls_fea['task_{}'.format(i)])
            _out = F.interpolate(_out, (H, W), mode='bilinear', align_corners=True)
            out.update({'task_{}'.format(i): _out})

        return out
