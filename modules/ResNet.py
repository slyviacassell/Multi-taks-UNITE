# coding=utf-8

'''
Created: 2021/3/20
@author: Slyviacassell@github.com
'''

import torch
import torch.nn as nn
from torchvision.models import resnet
from collections import OrderedDict
import os

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .Stage import Stage
from memonger import SublinearSequential


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-03, momentum=0.05, **kwargs):
        super(BatchNorm, self).__init__(num_features=num_features, eps=eps, momentum=momentum, **kwargs)


class ResNet(resnet.ResNet):
    def __init__(self, block, layers, is_head=True, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=BatchNorm):
        super(ResNet, self).__init__(block, layers, num_classes, zero_init_residual,
                                     groups, width_per_group, replace_stride_with_dilation, norm_layer)

        for p in self.avgpool.parameters():
            p.requires_grad_(False)
        for p in self.fc.parameters():
            p.requires_grad_(False)

        self.num_stages = 5
        pre_process = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)
        self.stages = nn.ModuleList()
        self.stages.append(Stage(64, pre_process))
        if block == resnet.Bottleneck:
            channels = 256
            out_fea_channels = 2048
        elif block == resnet.BasicBlock:
            channels = 64
            out_fea_channels = 512
        else:
            raise NotImplementedError
        for i in range(1, self.num_stages):
            stage = Stage(channels * pow(2, i - 1), getattr(self, 'layer' + str(i)))
            self.stages.append(stage)

        # simplified deeplab head
        self.is_head = is_head
        if self.is_head:
            head = [
                nn.Conv2d(out_fea_channels, 1024, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
                nn.BatchNorm2d(1024, eps=1e-03, momentum=0.05),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1024, eps=1e-03, momentum=0.05),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Conv2d(1024, num_classes, kernel_size=1)
            ]
            self.head = nn.Sequential(*head)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return SublinearSequential(*layers)

    def _init_head_weight_(self):
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, freeze=None):
        if freeze is not None:
            assert isinstance(freeze, int) or isinstance(freeze, list)
        for i in range(self.num_stages):
            if freeze is not None:
                if i in freeze:
                    self.stages[i].requires_grad_(False)
                x = self.stages[i](x)
            else:
                x = self.stages[i](x)
        if self.is_head:
            if freeze is not None:
                if 5 in freeze or 5 == freeze:
                    self.head.requires_grad_(False)
            x = self.head(x)
        return x


def _resnet(arch, block, layers, pretrained: bool, progress,
            weights_dir, weights: str = None, dataset: str = 'nyud',
            map_location=None,
            **kwargs):
    model = ResNet(block, layers, **kwargs)
    assert dataset in ['nyud', 'cityscapes', 'taskonomy', 'pascal_context']
    assert arch in ['resnet50', 'resnet101']
    assert weights in ['DeepLabV3', 'DeepLabV3Plus', 'Seg', 'Normal', None]

    if pretrained:
        state_dict = load_state_dict_from_url(resnet.model_urls[arch],
                                              progress=progress, map_location=map_location)
        matched_state_dict = OrderedDict()
        # remove the fully connected layers
        for k, v in state_dict.items():
            if 'fc' not in k:
                matched_state_dict.update({k: v})
        model.load_state_dict(matched_state_dict, strict=False)
    elif not pretrained and weights == 'DeepLabV3':
        state_dict = torch.load(os.path.join(weights_dir, 'deeplabv3/deeplabv3_{}_voc_os16.pth'.format(arch)),
                                map_location=map_location)
        matched_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'fc' not in k:
                matched_state_dict.update({k: v})
        model.load_state_dict(matched_state_dict, strict=False)
    elif not pretrained and weights == 'DeepLabV3Plus':
        state_dict = torch.load(os.path.join(weights_dir, 'deeplabv3/deeplabv3plus_{}_voc_os16.pth'.format(arch)),
                                map_location=map_location)
        matched_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'fc' not in k:
                matched_state_dict.update({k: v})
        model.load_state_dict(matched_state_dict, strict=False)

    # finetuned
    elif not pretrained and weights == 'Seg':
        state_dict = torch.load(os.path.join(weights_dir, dataset,
                                             'torch_finetune_seg_{}_os8.pth'.format(arch)),
                                map_location=map_location)
        model.load_state_dict(state_dict)
    elif not pretrained and weights == 'Normal':
        state_dict = torch.load(os.path.join(weights_dir, dataset,
                                             'torch_finetune_normal_{}_os8.pth'.format(arch)),
                                map_location=map_location)
        model.load_state_dict(state_dict)
    elif not pretrained and weights is None:
        print('[II] Random init resnet')
    else:
        raise NotImplementedError('[EE] Undefined weight {}'.format(weights))

    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', resnet.BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', resnet.BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if 'replace_stride_with_dilation' not in kwargs:
        return _resnet('resnet50', resnet.Bottleneck, [3, 4, 6, 3], pretrained, progress,
                       # the output stride is 16 https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/0c67dce524b2eb94dc3587ff2832e28f11440cae/network/modeling.py#L12
                       # here assume output stride == 8 for resnetfor resnet
                       replace_stride_with_dilation=[False, True, True], **kwargs)
        # output stride == 16
        # replace_stride_with_dilation=[False, False, True], **kwargs)
    else:
        return _resnet('resnet50', resnet.Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if 'replace_stride_with_dilation' not in kwargs:
        return _resnet('resnet101', resnet.Bottleneck, [3, 4, 23, 3], pretrained, progress,
                       # the output stride is 16 https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/0c67dce524b2eb94dc3587ff2832e28f11440cae/network/modeling.py#L12
                       # here assume output stride == 8 for resnet
                       replace_stride_with_dilation=[False, True, True], **kwargs)
        # output stride == 16
        # replace_stride_with_dilation=[False, False, True], **kwargs)
    else:
        return _resnet('resnet101', resnet.Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', resnet.Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', resnet.Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', resnet.Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
