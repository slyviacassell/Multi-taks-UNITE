# coding=utf-8

'''
Created: 2021/12/3
@author: Slyviacassell@github.com
'''

# pytorch
import torch
import torch.optim as optim
import torch.nn.functional as F

# determined
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from determined.pytorch import (
    PyTorchTrial,
    PyTorchTrialContext,
    LRScheduler,
    DataLoader
)

import numpy as np

# utils
from typing import (
    Any,
    Dict,
    Union,
    List,
    Sequence
)
import os

# costume
from data.nyud import NYUD
from modules.VGG16 import DeepLabLargeFOVBN
from modules.ResNet import resnet50, resnet101
from modules.DeepLabV3Plus import DeepLabV3, DeepLabV3Plus
from modules.UNITENet import UNITENet
from utils.get_scheduler import get_lr_scheduler
from utils.STLLoss import STLLoss
from utils.DensePredictionsReducer import DensePredictionsReducer
from experiments.stl import get_cfg_defaults
from utils.init_seed import init_seeds

from utils.metrics import (
    get_mIoU,
    get_conf_mat,
    get_normal_cosine,
    get_normal_metrics
)

# from utils.gpu_mem_track import MemTracker
# from pytorch_memlab import MemReporter

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class STLTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context

        # self.mem_tracker =MemTracker()

        # cannot be used determined --test
        self.det_exp_cfg = self.context.get_experiment_config()
        print(self.det_exp_cfg)

        cfg = get_cfg_defaults()
        cfg.merge_from_file(self.context.get_hparam('config'))
        cfg.freeze()
        self.cfg = cfg

        # print cfg
        print(cfg)

        self.last_epoch_idx = -1

        # tensorboard logging
        self.logger = TorchWriter()

        self.dataset_name = cfg.DATASET_NAME
        self.dataset_dir = cfg.DATASET_DIR

        # stl cfg
        self.task = cfg.STL.TASK
        self.n_output = cfg.STL.N_OUTPUT
        self.backbone_type = cfg.STL.BACKBONE_TYPE
        self.weight_dir = cfg.STL.WEIGHT_DIR
        self.backbone_weight = cfg.STL.BACKBONE_WEIGHT

        # optimization cfg
        self.base_lr = cfg.BASE_LR
        self.base_factor = cfg.BASE_FACTOR
        self.fc8_w_factor = cfg.FC8_W_FACTOR  # used for v1
        self.fc8_b_factor = cfg.FC8_B_FACTOR  # used for v1
        self.head_factor = cfg.HEAD_FACTOR  # used for v1
        self.classifier_factor = cfg.CLASSIFIER_FACTOR  # used for resnet

        self.weight_decay = cfg.WEIGHT_DECAY
        self.momentum = cfg.MOMENTUM

        self.n_train_units = self.det_exp_cfg['searcher']['max_length']['epochs'] * \
                             np.ceil(self.det_exp_cfg['records_per_epoch'] /
                                     self.context.get_global_batch_size())  # not available for non-local test
        self.n_warmup_units = cfg.N_WARMUP_UNITS
        assert self.n_warmup_units < self.n_train_units
        self.scheduler = cfg.SCHEDULER
        self.poly_power = cfg.POLY_POWER

        self.n_workers = self.context.get_hparam('n_workers')

        assert self.dataset_name in ['nyud', 'cityscapes', 'pascal_context', 'taskonomy']
        assert self.backbone_type in ['VGG16V1', 'R50V1', 'R50V3', 'R50V3P', 'R101V1', 'R101V3', 'R101V3P']

        # reproducibility
        # determined <= 0.7.5 is not able to set all used random seed
        # see determined\pytorch\_pytorch_trial.py#Line78 for details
        init_seeds(self.det_exp_cfg['reproducibility']['experiment_seed'])

        if self.dataset_name == 'nyud':
            self.train_datasets = NYUD(data_dir=self.dataset_dir,
                                       data_list_1=os.path.join(self.dataset_dir, 'list/training_seg.txt'),
                                       data_list_2=os.path.join(self.dataset_dir, 'list/training_normal_mask.txt'),
                                       output_size=self.cfg.TRAIN.OUTPUT_SIZE,
                                       random_scale=self.cfg.TRAIN.RANDOM_SCALE,
                                       random_mirror=self.cfg.TRAIN.RANDOM_MIRROR,
                                       random_crop=self.cfg.TRAIN.RANDOM_CROP,
                                       color_jitter=self.cfg.TRAIN.RANDOM_JITTER,
                                       ignore_label=self.cfg.IGNORE_LABEL)
            self.test_datasets = NYUD(data_dir=self.dataset_dir,
                                      data_list_1=os.path.join(self.dataset_dir, 'list/testing_seg.txt'),
                                      data_list_2=os.path.join(self.dataset_dir, 'list/testing_normal_mask.txt'),
                                      output_size=self.cfg.TEST.OUTPUT_SIZE,
                                      random_scale=self.cfg.TEST.RANDOM_SCALE,
                                      random_mirror=self.cfg.TEST.RANDOM_MIRROR,
                                      random_crop=self.cfg.TEST.RANDOM_CROP,
                                      color_jitter=self.cfg.TEST.RANDOM_JITTER,
                                      ignore_label=self.cfg.IGNORE_LABEL)
        else:
            raise NotImplementedError('[EE] Notimplemented datasets: {}'.format(self.dataset_name))

        is_deeplabv3 = False

        if self.backbone_type == 'VGG16V1':
            _backbone = DeepLabLargeFOVBN(in_dim=3,
                                          out_dim=self.n_output,
                                          weights_dir=self.weight_dir,
                                          weights=self.backbone_weight,
                                          dataset=self.dataset_name)
        elif self.backbone_type == 'R50V1':
            _backbone = resnet50(pretrained=False,
                                 weights_dir=self.weight_dir,
                                 weights=self.backbone_weight,
                                 num_classes=self.n_output,
                                 dataset=self.dataset_name,
                                 # os=16
                                 # replace_stride_with_dilation=[False, False, True],
                                 # os=8
                                 replace_stride_with_dilation=[False, True, True],
                                 )
        elif self.backbone_type == 'R101V1':
            _backbone = resnet101(pretrained=False,
                                  weights_dir=self.weight_dir,
                                  weights=self.backbone_weight,
                                  num_classes=self.n_output,
                                  dataset=self.dataset_name,
                                  # os=16
                                  # replace_stride_with_dilation=[False, False, True],
                                  # os=8
                                  replace_stride_with_dilation=[False, True, True],
                                  )
        elif self.backbone_type == 'R50V3':
            _backbone = DeepLabV3(backbone_name='resnet50',
                                  pretrained=False,
                                  weights_dir=self.weight_dir,
                                  weight=self.backbone_weight,
                                  num_classes=self.n_output,
                                  # os=16
                                  replace_stride_with_dilation=[False, False, True],
                                  aspp_dilated=[6, 12, 18],
                                  # os=8
                                  # replace_stride_with_dilation=[False, True, True],
                                  # aspp_dilated=[12, 24, 36]
                                  )
            is_deeplabv3 = True
        elif self.backbone_type == 'R101V3':
            _backbone = DeepLabV3(backbone_name='resnet101',
                                  pretrained=False,
                                  weights_dir=self.weight_dir,
                                  weight=self.backbone_weight,
                                  num_classes=self.n_output,
                                  # os=16
                                  replace_stride_with_dilation=[False, False, True],
                                  aspp_dilated=[6, 12, 18],
                                  # os=8
                                  # replace_stride_with_dilation=[False, True, True],
                                  # aspp_dilated=[12, 24, 36]
                                  )
            is_deeplabv3 = True
        elif self.backbone_type == 'R50V3P':
            _backbone = DeepLabV3Plus(backbone_name='resnet50',
                                      pretrained=False,
                                      weights_dir=self.weight_dir,
                                      weight=self.backbone_weight,
                                      num_classes=self.n_output,
                                      # os=16
                                      replace_stride_with_dilation=[False, False, True],
                                      aspp_dilated=[6, 12, 18],
                                      # os=8
                                      # replace_stride_with_dilation=[False, True, True],
                                      # aspp_dilated=[12, 24, 36]
                                      )
            is_deeplabv3 = True
        elif self.backbone_type == 'R101V3P':
            _backbone = DeepLabV3Plus(backbone_name='resnet101',
                                      pretrained=False,
                                      weights_dir=self.weight_dir,
                                      weight=self.backbone_weight,
                                      num_classes=self.n_output,
                                      # os=16
                                      replace_stride_with_dilation=[False, False, True],
                                      aspp_dilated=[6, 12, 18],
                                      # os=8
                                      # replace_stride_with_dilation=[False, True, True],
                                      # aspp_dilated=[12, 24, 36]
                                      )
            is_deeplabv3 = True
        else:
            raise NotImplementedError('[EE] Unimplemented backbone: {}'.format(self.backbone_type))

        self.model = self.context.wrap_model(_backbone)

        no_decay = []
        base_param = []
        base_nd = []
        fc8_weights = []
        fc8_bias = []
        head_param = []
        head_nd = []
        classifier_params = []
        classifier_nd = []

        for k, v in self.model.named_parameters():
            if 'head' in k:
                if 'head.8' in k:
                    if 'weight' in k:
                        fc8_weights.append(v)
                    else:
                        assert 'bias' in k
                        fc8_bias.append(v)
                else:
                    if any(nd in k for nd in no_decay):
                        head_nd.append(v)
                    else:
                        head_param.append(v)
            # only v3 and v3p use this
            elif 'classifier' in k:
                if any(nd in k for nd in no_decay):
                    classifier_nd.append(v)
                else:
                    classifier_params.append(v)
            else:
                # filter out fc
                if 'fc' in k:
                    print('skip fc param with name {}'.format(k))
                    continue

                if any(nd in k for nd in no_decay):
                    base_nd.append(v)
                else:
                    base_param.append(v)

        optimizer_grouped_parameters = [
            {'params': base_param,
             'lr': self.base_lr * self.base_factor},
        ]
        # for onecycle
        max_lrs=[self.base_lr * self.base_factor,]


        if len(no_decay) >0:
            optimizer_grouped_parameters+=[
                {'params': base_nd,
             'lr': self.base_lr * self.base_factor,
             'weight_decay': 0.0},
            ]
            max_lrs+=[self.base_lr * self.base_factor,]

        # assume that only R50/R101V1
        if 'V1' in self.backbone_type:
            optimizer_grouped_parameters += [
                {'params': head_param,
                 'lr': self.base_lr * self.head_factor},
                {'params': fc8_weights,
                     'lr': self.base_lr * self.fc8_w_factor},
                {'params': fc8_bias,
                     'lr': self.base_lr * self.fc8_b_factor,
                     'weight_decay': 0.0 if len(no_decay) > 0 else self.weight_decay},
            ]
            max_lrs += [
                        self.base_lr * self.head_factor,
                        self.base_lr * self.fc8_w_factor,
                        self.base_lr * self.fc8_b_factor
                        ]

            if len(no_decay) > 0:
                optimizer_grouped_parameters +=[
                    {'params': head_nd,
                     'lr': self.base_lr * self.head_factor,
                     'weight_decay': 0.0},
                ]
                max_lrs += [self.base_lr * self.head_factor, ]

        elif 'V3' in self.backbone_type:
            optimizer_grouped_parameters += [
                {'params': classifier_params,
                 'lr': self.base_lr * self.classifier_factor},
            ]
            max_lrs += [self.base_lr * self.classifier_factor, ]

            if len(no_decay) > 0:
                optimizer_grouped_parameters +=[
                    {'params': classifier_nd,
                     'lr': self.base_lr * self.head_factor,
                     'weight_decay': 0.0},
                ]
                max_lrs += [self.base_lr * self.classifier_factor, ]

        self.optimizer = self.context.wrap_optimizer(optim.SGD(optimizer_grouped_parameters,
                                                               lr=self.base_lr,
                                                               weight_decay=self.weight_decay,
                                                               momentum=self.momentum))

        self.lr_scheduler = self.context.wrap_lr_scheduler(
            optim.lr_scheduler.OneCycleLR(self.optimizer,
                                          # need to be same with the num of parameter groups
                                          max_lr=max_lrs,
                                          total_steps=self.det_exp_cfg['searcher']['max_length']['epochs']
                                          ),
            LRScheduler.StepMode.STEP_EVERY_EPOCH,
            frequency=1)

        self.loss_func = STLLoss(self.task)

        # self.training_reducer = self.context.wrap_reducer(MTLPredictionsReducer(self.tasks, is_training=True),
        #                                                   for_training=True, for_validation=False)

    def build_training_data_loader(self) -> DataLoader:
        return DataLoader(self.train_datasets, batch_size=self.context.get_per_slot_batch_size(),
                          shuffle=True,
                          num_workers=self.n_workers, pin_memory=True)

    def build_validation_data_loader(self) -> DataLoader:
        return DataLoader(self.test_datasets, batch_size=self.context.get_per_slot_batch_size(),
                          shuffle=True,
                          num_workers=self.n_workers, pin_memory=True)

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, Any]:
        img = batch[0]

        if self.task=='seg':
            labels=batch[1].squeeze(1)
            out_sz=labels.size()[1:]
        elif self.task=='normal':
            labels=batch[2]
            out_sz=labels.size()[2:]

        if 'V3' in self.backbone_type:
            out = self.model(img)['out']
        else:
            out = self.model(img)
        out = F.interpolate(out, out_sz, align_corners=True, mode='bilinear')
        loss = self.loss_func(out, labels)

        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        # the loss will be mean-reduced across the epoch
        return {self.task:loss.detach().item()}

    def evaluate_full_dataset(self, data_loader: DataLoader) -> Dict[str, Any]:
        # only evaluate model with single slot
        # fixme: currently only implemented for normal, segmentation
        if 'normal' == self.task:
            cosine = []  # type: List[np.array]
            cosine_mask = []  # type: List[np.array]

        if 'seg' == self.task:
            conf_mat_sum = 0

        n_batch = len(data_loader)
        batch_metrics = []  # type: List[Dict]
        for _, batch in enumerate(data_loader):

            img = self.context.to_device(batch[0])
            if self.task == 'seg':
                labels = self.context.to_device(batch[1]).squeeze(1)
                out_sz=labels.size()[1:]
            elif self.task == 'normal':
                labels = self.context.to_device(batch[2])
                out_sz=labels.size()[2:]

            if 'V3' in self.backbone_type:
                out = self.model(img)['out']
            else:
                out = self.model(img)
            out = F.interpolate(out, out_sz, align_corners=True, mode='bilinear')
            loss = self.loss_func(out, labels)

            _metrics = {}
            _metrics.update({'validation_loss': loss.detach().item()})
            batch_metrics.append(_metrics)

            if self.task == 'seg':
                # fixme: make the num of classes be configurable
                _conf_mat = get_conf_mat(out, labels, 40)
                conf_mat_sum += _conf_mat.cpu().numpy()
            elif self.task == 'normal':
                _cosine, _mask = get_normal_cosine(prediction=out,
                                                   gt=labels, normalization=False,
                                                   ignore_label=255)
                cosine.append(_cosine.cpu().numpy())
                cosine_mask.append(_mask.cpu().numpy())
            else:
                raise NotImplementedError('[EE] Unimplemented tasks: {}'.format(self.task))

        # statistics
        _reduced_metrics = {}

        _reduced_metrics.update({'validation_loss': sum([p['validation_loss'] for p in batch_metrics]) / n_batch})

        if 'normal' == self.task:
            _normal_loss = sum([p['validation_loss'] for p in batch_metrics]) / n_batch
            _reduced_metrics.update({'normal': _normal_loss})

            _cos_angels = [c[m] for c, m in zip(cosine, cosine_mask)]
            normal_metrics = get_normal_metrics(_cos_angels)

            _reduced_metrics.update({**normal_metrics})

        if 'seg' == self.task:
            _seg_loss = sum([p['validation_loss'] for p in batch_metrics]) / n_batch
            _reduced_metrics.update({'seg': _seg_loss})

            _mIoU = get_mIoU(conf_mat_sum) * 100.
            _pixel_acc = np.diag(conf_mat_sum).sum() / np.sum(conf_mat_sum) * 100.
            _reduced_metrics.update({'mIoU': _mIoU, 'pixel_acc': _pixel_acc})

        return _reduced_metrics
