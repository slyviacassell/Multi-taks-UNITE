# coding=utf-8

'''
Created: 2021/3/1
@author: Slyviacassell@github.com
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import (
    Dict
)
from collections import OrderedDict

from .UNITELayer import UNITELayer
from .DeepLabV3Plus import DeepLabV3, DeepLabV3Plus
from .ResNet import ResNet
from .VGG16 import DeepLabLargeFOVBN

# ablation
from .DoubleLayer import DoubleLayer


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
                 similarity='bilinear',
                 is_deeplabv3: bool=False):
        super(UNITENet, self).__init__()
        assert len(backbones) == n_tasks
        self.n_tasks = n_tasks
        self.backbones = nn.ModuleDict(backbones)

        self.is_deeplabv3 = is_deeplabv3

        # assume that all backbones have same num of stages and same architectures
        for k, b in backbones.items():
            if is_deeplabv3:
                pass
            else:
                assert len(n_patch) == len(b.stages)

        self.num_stages = len(n_patch)
        self.n_patch = n_patch
        unites = []
        total_channels = 0

        net1 = backbones['task_0']
        # be adaptive for deeplab
        if isinstance(net1, (DeepLabV3, DeepLabV3Plus)):
            net1 = net1.backbone
            self.is_deeplabv3 = True

        for stage_id in range(self.num_stages):
            out_channels = net1.stages[stage_id].out_channels
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

            # ablation
            # _unite = DoubleLayer(out_channels=out_channels,
            #                      init_weights=init_weights,
            #                      init_method=init_method,
            #                      activation=activation,
            #                      batch_norm=batch_norm,
            #                      bn_before_activation=bn_before_activation,
            #                      conv_bias=conv_bias,
            #                      n_tasks=n_tasks)

            unites.append(_unite)

        unites = nn.ModuleList(unites)

        self.shortcut = shortcut
        final_conv = []

        if shortcut:
            print("Using multi-scale")
            for i in range(self.n_tasks):
                conv = nn.Conv2d(total_channels, net1.stages[-1].out_channels, kernel_size=1, bias=conv_bias)
                bn = nn.BatchNorm2d(net1.stages[-1].out_channels, eps=1e-03, momentum=0.05)

                if bn_before_activation:
                    print("Using bn before activation")
                    _final_conv = [conv, bn, nn.ReLU()]
                else:
                    _final_conv = [conv, nn.ReLU(), bn]

                _final_conv = nn.Sequential(*_final_conv)

                final_conv.append(_final_conv)

        self.unites = nn.ModuleDict({
            'unites': unites,
            'shortcut': nn.ModuleList(final_conv) if self.shortcut else None,
        })

    def forward(self, features: Dict):
        # assume all features have same size
        _, _, H, W = features['task_0'].size()

        task_shortcuts = []
        out = {**features}

        low_feas=None

        # unite_loss = 0.
        for stage_id in range(self.num_stages):
            if self.is_deeplabv3:
                out.update({k: self.backbones[k].backbone.stages[stage_id](v) for k, v in out.items()})
            else:
                out.update({k: self.backbones[k].stages[stage_id](v) for k, v in out.items()})

            # define conv2 as low_level for deeplabv3p
            if self.is_deeplabv3 and isinstance(self.backbones['task_0'], DeepLabV3Plus) and stage_id==1:
                low_feas={**out}

            out = self.unites['unites'][stage_id](out)

            if self.shortcut:
                task_shortcuts.append({k: v for k, v in out.items()})
        if self.shortcut:
            _, _, h, w = task_shortcuts[-1]['task_0'].size()
            for i in range(self.n_tasks):
                _out = torch.cat(
                    [F.interpolate(_x['task_{}'.format(i)], (h, w), mode='bilinear', align_corners=True) for _x in
                     task_shortcuts[:-1]] + [task_shortcuts[-1]['task_{}'.format(i)]], dim=1)

                # using correspondence shortcut
                _out = self.unites['shortcut'][i](_out)

                out.update({'task_{}'.format(i): _out})

        for i in range(self.n_tasks):
            if self.is_deeplabv3:
                if isinstance(self.backbones['task_{}'.format(i)], DeepLabV3Plus):
                    cls_feas={'low_level':low_feas['task_{}'.format(i)],'out':out['task_{}'.format(i)]}
                else:
                    cls_feas={'out':out['task_{}'.format(i)]}
                _out=self.backbones['task_{}'.format(i)].classifier(cls_feas)
            else:
                _out = self.backbones['task_{}'.format(i)].head(out['task_{}'.format(i)])

            _out = F.interpolate(_out, (H, W), mode='bilinear', align_corners=True)
            out.update({'task_{}'.format(i): _out})

        return out
