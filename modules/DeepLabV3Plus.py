# coding=utf-8

'''
Created: 2021/4/30
@author: Slyviacassell@github.com
'''

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from collections import OrderedDict
from torchvision.models._utils import IntermediateLayerGetter
import os

from .ResNet import resnet50, resnet101
from .VGG16 import DeepLabLargeFOVBN


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Arguments:
        backbone_name (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    def __init__(self, backbone_name: str, pretrained: bool, weights_dir: str, weight, num_classes,
                 map_location=None, replace_stride_with_dilation=[False, False, True], aspp_dilated=[6, 12, 18],
                 aux_classifier=None):
        # output stride == 8
        # replace_stride_with_dilation=[False, True, True], aspp_dilated=[12, 24, 36]
        assert backbone_name in ['resnet50', 'resnet101']
        self.weights_dir = weights_dir

        if backbone_name == 'resnet50':
            backbone = resnet50(pretrained=pretrained, weights=weight, num_classes=num_classes,
                                map_location=map_location,
                                weights_dir=weights_dir,
                                replace_stride_with_dilation=replace_stride_with_dilation, is_head=False)
        elif backbone_name == 'resnet101':
            backbone = resnet101(pretrained=pretrained, weights=weight, num_classes=num_classes,
                                 map_location=map_location,
                                 weights_dir=weights_dir,
                                 replace_stride_with_dilation=replace_stride_with_dilation, is_head=False)

        classifier = DeepLabHead(in_channels=backbone.stages[-1].out_channels,
                                 num_classes=num_classes, aspp_dilate=aspp_dilated)
        super(DeepLabV3, self).__init__(backbone, classifier, aux_classifier)
        if not pretrained:
            self._init_weight(backbone_name, weight, map_location=map_location)

    def _init_weight(self, backbone_name, weight, dataset=None, map_location=None):
        assert weight in ['DeepLabV3', 'DeepLabV3Plus']
        assert backbone_name in ['resnet50', 'resnet101']

        if weight == 'DeepLabV3':
            model_dict = self.state_dict()
            pretrained_dict = torch.load(os.path.join(self.weights_dir, 'deeplabv3',
                                                      'seg_deeplabv3_{}_voc_os16.pth'.format(backbone_name)),
                                         map_location=map_location)
            model_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'classifier.4' not in k}
            self.load_state_dict(model_dict, strict=False)
        elif weight == 'DeepLabV3Plus':
            model_dict = self.state_dict()
            pretrained_dict = torch.load(os.path.join(self.weights_dir, 'deeplabv3',
                                                      'seg_deeplabv3plus_{}_voc_os16.pth'.format(backbone_name)),
                                         map_location=map_location)
            model_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'classifier.3' not in k}
            self.load_state_dict(model_dict, strict=False)

        # finetuned
        elif weight == 'DeepLabV3Plus-Seg':
            pretrained_dict = torch.load(os.path.join(self.weights_dir, dataset,
                                                      'finetune_seg_deeplabv3plus_{}_voc_os16.pth'.format(
                                                          backbone_name)),
                                         map_location=map_location)
            self.load_state_dict(pretrained_dict)
        elif weight == 'DeepLabV3Plus-Normal':
            pretrained_dict = torch.load(os.path.join(self.weights_dir, dataset,
                                                      'finetune_normal_deeplabv3plus_{}_voc_os16.pth'.format(
                                                          backbone_name)),
                                         map_location=map_location)
            self.load_state_dict(pretrained_dict)
        elif weight == 'DeepLabV3-Seg':
            pretrained_dict = torch.load(os.path.join(self.weights_dir, dataset,
                                                      'finetune_seg_deeplabv3_{}_voc_os16.pth'.format(backbone_name)),
                                         map_location=map_location)
            self.load_state_dict(pretrained_dict)
        elif weight == 'DeepLabV3-Normal':
            pretrained_dict = torch.load(os.path.join(self.weights_dir, dataset,
                                                      'finetune_normal_deeplabv3_{}_voc_os16.pth'.format(
                                                          backbone_name)),
                                         map_location=map_location)
            self.load_state_dict(pretrained_dict)

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors

        features = OrderedDict()
        features['out'] = self.backbone(x)

        result = OrderedDict()
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        result["out"] = x

        if self.aux_classifier is not None:
            assert 'aux' in features
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            result["aux"] = x

        return result


class DeepLabV3Plus(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Arguments:
        backbone_name (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    def __init__(self, backbone_name: str, pretrained, weights_dir: str, weight, num_classes,
                 map_location=None, replace_stride_with_dilation=[False, False, True], aspp_dilated=[6, 12, 18],
                 aux_classifier=None):
        assert backbone_name in ['resnet50', 'resnet101']
        self.weights_dir = weights_dir

        if backbone_name == 'resnet50':
            backbone = resnet50(pretrained=pretrained, weights=weight, num_classes=num_classes,
                                map_location=map_location,
                                weights_dir=weights_dir,
                                replace_stride_with_dilation=replace_stride_with_dilation, is_head=False)
        elif backbone_name == 'resnet101':
            backbone = resnet101(pretrained=pretrained, weights=weight, num_classes=num_classes,
                                 map_location=map_location,
                                 weights_dir=weights_dir,
                                 replace_stride_with_dilation=replace_stride_with_dilation, is_head=False)

        # conv2 as low_level features
        classifier = DeepLabHeadV3Plus(in_channels=backbone.stages[-1].out_channels,
                                       low_level_channels=backbone.stages[1].out_channels,
                                       num_classes=num_classes, aspp_dilate=aspp_dilated)
        super(DeepLabV3Plus, self).__init__(backbone, classifier, aux_classifier)
        if not pretrained:
            self._init_weight(backbone_name, weight, map_location=map_location)

    def _init_weight(self, backbone_name, weight, dataset=None, map_location=None):
        assert weight in ['DeepLabV3', 'DeepLabV3Plus']
        assert backbone_name in ['resnet50', 'resnet101']

        if weight == 'DeepLabV3':
            model_dict = self.state_dict()
            pretrained_dict = torch.load(os.path.join(self.weights_dir, 'deeplabv3',
                                                      'seg_deeplabv3_{}_voc_os16.pth'.format(backbone_name)),
                                         map_location=map_location)
            model_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'classifier.4' not in k}
            self.load_state_dict(model_dict, strict=False)
        elif weight == 'DeepLabV3Plus':
            model_dict = self.state_dict()
            pretrained_dict = torch.load(os.path.join(self.weights_dir, 'deeplabv3',
                                                      'seg_deeplabv3plus_{}_voc_os16.pth'.format(backbone_name)),
                                         map_location=map_location)
            model_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'classifier.3' not in k}
            self.load_state_dict(model_dict, strict=False)

        # finetuned
        elif weight == 'DeepLabV3Plus-Seg':
            pretrained_dict = torch.load(os.path.join(self.weights_dir, dataset,
                                                      'finetune_seg_deeplabv3plus_{}_voc_os16.pth'.format(
                                                          backbone_name)),
                                         map_location=map_location)
            self.load_state_dict(pretrained_dict)
        elif weight == 'DeepLabV3Plus-Normal':
            pretrained_dict = torch.load(os.path.join(self.weights_dir, dataset,
                                                      'finetune_normal_deeplabv3plus_resnet101_voc_os16.pth'.format(
                                                          backbone_name)),
                                         map_location=map_location)
            self.load_state_dict(pretrained_dict)
        elif weight == 'DeepLabV3-Seg':
            pretrained_dict = torch.load(os.path.join(self.weights_dir, dataset,
                                                      'finetune_seg_deeplabv3_{}_voc_os16.pth'.format(backbone_name)),
                                         map_location=map_location)
            self.load_state_dict(pretrained_dict)
        elif weight == 'DeepLabV3-Normal':
            pretrained_dict = torch.load(os.path.join(self.weights_dir, dataset,
                                                      'finetune_normal_deeplabv3_{}_voc_os16.pth'.format(
                                                          backbone_name)),
                                         map_location=map_location)
            self.load_state_dict(pretrained_dict)

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors

        features = OrderedDict()
        for stage_id in range(self.backbone.num_stages):
            x = self.backbone.stages[stage_id](x)
            # conv2 as low_level features
            if stage_id == 1:
                features['low_level'] = x
        features['out'] = x

        result = OrderedDict()
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
        result["out"] = x

        if self.aux_classifier is not None:
            assert 'aux' in features
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)
            result["aux"] = x

        return result


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[6, 12, 18]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48, eps=1e-3, momentum=0.05),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature['out'])

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.05),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.05),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module
