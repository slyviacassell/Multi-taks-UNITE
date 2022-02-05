# coding=utf-8

'''
Created: 2021/1/31
@author: Slyviacassell@github.com
'''

import torch
import torch.nn as nn
import os

from .Stage import Stage


class DeepLabLargeFOVBN(nn.Module):
    def __init__(self, in_dim, out_dim, weights_dir: str, weights='DeepLab', dataset: str = 'nyud', map_location=None):
        super(DeepLabLargeFOVBN, self).__init__()
        self.out_dim = out_dim
        self.weights_dir = weights_dir
        self.stages = []
        layers = []

        stage = [
            nn.Conv2d(in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.ConstantPad2d((0, 1, 0, 1), 0),  # TensorFlow 'SAME' behavior
            nn.MaxPool2d(3, stride=2)
        ]
        layers += stage
        self.stages.append(Stage(64, stage))

        stage = [
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.ConstantPad2d((0, 1, 0, 1), 0),  # TensorFlow 'SAME' behavior
            nn.MaxPool2d(3, stride=2)
        ]
        layers += stage
        self.stages.append(Stage(128, stage))

        stage = [
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.ConstantPad2d((0, 1, 0, 1), 0),  # TensorFlow 'SAME' behavior
            nn.MaxPool2d(3, stride=2)
        ]
        layers += stage
        self.stages.append(Stage(256, stage))

        stage = [
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1, padding=1)
        ]
        layers += stage
        self.stages.append(Stage(512, stage))

        stage = [
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=1, padding=1),
            # must use count_include_pad=False to make sure result is same as TF
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        ]
        layers += stage
        self.stages.append(Stage(512, stage))
        self.stages = nn.ModuleList(self.stages)

        # Used for backward compatibility with weight loading
        self.features = nn.Sequential(*layers)

        head = [
            # nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(1024, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024, eps=1e-03, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(1024, out_dim, kernel_size=1)
        ]
        self.head = nn.Sequential(*head)

        self.weights = weights
        self.init_weights(dataset, map_location)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        return x

    def init_weights(self, dataset: str, map_location=None):
        assert dataset in ['nyud', 'taskonomy', 'cityscapes', 'pascal']
        for layer in self.head.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=1)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

        if self.weights == 'DeepLab':
            pretrained_dict = torch.load(os.path.join(self.weights_dir, 'vgg_deeplab_lfov/tf_deeplab.pth'),
                                         map_location=map_location)
            model_dict = self.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and 'head.8' not in k}

            # # filter out the mismatched weights
            # mismatched_dict = {k: v for k, v in model_dict.items() if
            #                    k in pretrained_dict and v.size() != pretrained_dict[k].size()}
            #
            # # init and update mismatched weights
            # for k, v in mismatched_dict.items():
            #     nn.init.kaiming_normal_(v)
            # pretrained_dict.update(mismatched_dict)

            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.load_state_dict(model_dict)
        elif self.weights == 'Seg':
            pretrained_dict = torch.load(os.path.join(self.weights_dir, dataset, 'tf_finetune_seg.pth'),
                                         map_location=map_location)
            self.load_state_dict(pretrained_dict)
        elif self.weights == 'Normal':
            pretrained_dict = torch.load(os.path.join(self.weights_dir, dataset, 'tf_finetune_normal.pth'),
                                         map_location=map_location)
            self.load_state_dict(pretrained_dict)
        else:
            raise NotImplementedError
