# coding=utf-8

'''
Created: 2021/12/3
@author: Slyviacassell@github.com
'''

# pytorch
import torch
import torch.optim as optim

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
from utils.MTLLosses import MTLLosses
from utils.MTLPredictionsReducer import MTLPredictionsReducer
from experiments.mtl import get_cfg_defaults
from optimizer.adabound import AdaBound, AdaBoundW
from utils.init_seed import init_seeds

from utils.metrics import (
    get_mIoU,
    get_conf_mat,
    get_normal_cosine,
    get_normal_metrics
)
from modules.test.NDDRNetAttention import NDDRNetAttention

# from utils.gpu_mem_track import MemTracker
# from pytorch_memlab import MemReporter

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class UNITETrainEvalTrail(PyTorchTrial):
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

        # mlt configs
        self.tasks = cfg.MTL.TASKS
        self.n_tasks = len(self.tasks)
        self.n_output_per_task = cfg.MTL.N_OUTPUT_PER_TASK
        self.backbone_type = cfg.MTL.BACKBONE_TYPE
        self.weight_dir = cfg.MTL.WEIGHT_DIR
        self.weight_per_task = cfg.MTL.WEIGHT_PER_TASK
        self.loss_factors = cfg.MTL.LOSS_FACTORS

        # unite cfg
        self.n_patches = cfg.MODEL.N_PATCHES
        self.similarity = cfg.MODEL.SIMILARITY
        self.reconstruct_src = cfg.MODEL.RECONSTRUCT_SRC
        self.init_method = cfg.MODEL.INIT_METHOD
        self.init_weights = cfg.MODEL.INIT_WEIGHTS
        self.shortcut = cfg.MODEL.SHORTCUT
        self.bn_before_activation = cfg.MODEL.BN_BEFORE_ACTIVATION

        # optimization cfg
        self.base_lr = cfg.BASE_LR
        self.base_factor = cfg.BASE_FACTOR
        self.unites_factor = cfg.UNITES_FACTOR
        self.fc8_w_factor = cfg.FC8_W_FACTOR  # used for vgg
        self.fc8_b_factor = cfg.FC8_B_FACTOR  # used for vgg
        self.head_factor = cfg.HEAD_FACTOR  # used for r50v1/r101v1
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

        assert isinstance(self.n_output_per_task, list) and len(self.n_output_per_task) == self.n_tasks
        # assume that the order of # of outputs is along with taht of weight_per_task
        assert isinstance(self.weight_per_task, list) and len(self.weight_per_task) == self.n_tasks
        assert self.dataset_name in ['nyud', 'cityscapes', 'pascal_context', 'taskonomy']
        assert self.backbone_type in ['VGG16V1', 'R50V1', 'R50V3', 'R50V3P', 'R101V1', 'R101V3', 'R101V3P']
        assert isinstance(self.n_patches, list)
        assert isinstance(self.init_weights, list)

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

        # multi-task
        if self.n_tasks > 1:
            backbones = {}
            is_deeplabv3 = False
            for i in range(self.n_tasks):
                if self.backbone_type == 'VGG16V1':
                    _backbone = DeepLabLargeFOVBN(in_dim=3,
                                                  out_dim=self.n_output_per_task[i],
                                                  weights_dir=self.weight_dir,
                                                  weights=self.weight_per_task[i],
                                                  dataset=self.dataset_name)
                    backbones.update({'task_{}'.format(i): _backbone})
                elif self.backbone_type == 'R50V1':
                    _backbone = resnet50(pretrained=False,
                                         weights_dir=self.weight_dir,
                                         weights=self.weight_per_task[i],
                                         num_classes=self.n_output_per_task[i],
                                         dataset=self.dataset_name,
                                         # os=16
                                         # replace_stride_with_dilation=[False, False, True],
                                         # os=8
                                         replace_stride_with_dilation=[False, True, True],
                                         )
                    backbones.update({'task_{}'.format(i): _backbone})
                elif self.backbone_type == 'R101V1':
                    _backbone = resnet101(pretrained=False,
                                          weights_dir=self.weight_dir,
                                          weights=self.weight_per_task[i],
                                          num_classes=self.n_output_per_task[i],
                                          dataset=self.dataset_name,
                                          # os=16
                                          # replace_stride_with_dilation=[False, False, True],
                                          # os=8
                                          replace_stride_with_dilation=[False, True, True],
                                          )
                    backbones.update({'task_{}'.format(i): _backbone})
                elif self.backbone_type == 'R50V3':
                    _backbone = DeepLabV3(backbone_name='resnet50',
                                          pretrained=False,
                                          weights_dir=self.weight_dir,
                                          weight=self.weight_per_task[i],
                                          num_classes=self.n_output_per_task[i],
                                          # os=16
                                          # replace_stride_with_dilation=[False, False, True],
                                          # aspp_dilated=[6, 12, 18],
                                          # os=8
                                          replace_stride_with_dilation=[False, True, True],
                                          aspp_dilated=[12, 24, 36]
                                          )
                    backbones.update({'task_{}'.format(i): _backbone})
                    is_deeplabv3=True
                elif self.backbone_type == 'R101V3':
                    _backbone = DeepLabV3(backbone_name='resnet101',
                                          pretrained=False,
                                          weights_dir=self.weight_dir,
                                          weight=self.weight_per_task[i],
                                          num_classes=self.n_output_per_task[i],
                                          # os=16
                                          # replace_stride_with_dilation=[False, False, True],
                                          # aspp_dilated=[6, 12, 18],
                                          # os=8
                                          replace_stride_with_dilation=[False, True, True],
                                          aspp_dilated=[12, 24, 36]
                                          )
                    backbones.update({'task_{}'.format(i): _backbone})
                    is_deeplabv3=True
                elif self.backbone_type == 'R50V3P':
                    _backbone = DeepLabV3Plus(backbone_name='resnet50',
                                              pretrained=False,
                                              weights_dir=self.weight_dir,
                                              weight=self.weight_per_task[i],
                                              num_classes=self.n_output_per_task[i],
                                              # os=16
                                              # replace_stride_with_dilation=[False, False, True],
                                              # aspp_dilated=[6, 12, 18],
                                              # os=8
                                              replace_stride_with_dilation=[False, True, True],
                                              aspp_dilated=[12, 24, 36]
                                              )
                    backbones.update({'task_{}'.format(i): _backbone})
                    is_deeplabv3=True
                elif self.backbone_type == 'R101V3P':
                    _backbone = DeepLabV3Plus(backbone_name='resnet101',
                                              pretrained=False,
                                              weights_dir=self.weight_dir,
                                              weight=self.weight_per_task[i],
                                              num_classes=self.n_output_per_task[i],
                                              # os=16
                                              # replace_stride_with_dilation=[False, False, True],
                                              # aspp_dilated=[6, 12, 18],
                                              # os=8
                                              replace_stride_with_dilation=[False, True, True],
                                              aspp_dilated=[12, 24, 36]
                                              )
                    backbones.update({'task_{}'.format(i): _backbone})
                    is_deeplabv3=True
                else:
                    raise NotImplementedError('[EE] Unimplemented backbone: {}'.format(self.backbone_type))
            _mtl_net = UNITENet(backbones=backbones,
                                n_tasks=self.n_tasks,
                                n_patch=self.n_patches,
                                reconstruct_src=self.reconstruct_src,
                                init_weights=self.init_weights,
                                init_method=self.init_method,
                                shortcut=self.shortcut,
                                bn_before_activation=self.bn_before_activation,
                                similarity=self.similarity,
                                is_deeplabv3=is_deeplabv3)

            # self.mem_tracker.track()
            self.model = self.context.wrap_model(_mtl_net)
            # self.mem_tracker.track()

            # self.reporter = MemReporter(self.model)
            # print('========= after to =========')
            # self.reporter.report(verbose=True)

            no_decay = []
            base_param = []
            base_nd = []
            fc8_weights = []
            fc8_bias = []

            head_param=[]
            head_nd=[]

            unites_param = []
            unites_nd = []
            classifier_params = []
            classifier_nd = []

            for k, v in self.model.named_parameters():
                if 'unites' in k:
                    if any(nd in k for nd in no_decay):
                        unites_nd.append(v)
                    else:
                        unites_param.append(v)
                elif 'head' in k:
                # not head
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
                {'params': unites_param,
                 'lr': self.base_lr * self.unites_factor},
            ]

            # for onecos
            max_lrs=[
                self.base_lr * self.base_factor,
                self.base_lr * self.unites_factor,
            ]

            if len(no_decay)>0:
                optimizer_grouped_parameters+=[
                    {'params': base_nd,
                     'lr': self.base_lr * self.base_factor,
                     'weight_decay': 0.0},
                    {'params': unites_nd,
                     'lr': self.base_lr * self.unites_factor,
                     'weight_decay': 0.0},
                ]
                max_lrs+=[
                    self.base_lr * self.base_factor,
                    self.base_lr * self.unites_factor,
                ]

            if 'V1' in self.backbone_type:
                optimizer_grouped_parameters += [
                    {'params': fc8_weights,
                     'lr': self.base_lr * self.fc8_w_factor},
                    {'params': fc8_bias,
                     'lr': self.base_lr * self.fc8_b_factor,
                     'weight_decay': 0.0 if len(no_decay)>0 else self.weight_decay},
                ]
                max_lrs += [
                    self.base_lr * self.fc8_w_factor,
                    self.base_lr * self.fc8_b_factor,
                ]

                optimizer_grouped_parameters += [
                    {'params': head_param,
                     'lr': self.base_lr * self.head_factor},
                ]
                max_lrs += [
                    self.base_lr * self.head_factor,
                ]

                if len(no_decay) > 0:
                    optimizer_grouped_parameters += [
                        {'params': head_nd,
                         'lr': self.base_lr * self.head_factor,
                         'weight_decay': 0.0},
                    ]
                    max_lrs += [
                        self.base_lr * self.head_factor,
                    ]

            elif 'V3' in self.backbone_type:
                optimizer_grouped_parameters += [
                    {'params': classifier_params,
                     'lr': self.base_lr * self.classifier_factor},
                ]
                max_lrs += [
                    self.base_lr * self.classifier_factor,
                ]

                if len(no_decay) > 0:
                    optimizer_grouped_parameters += [
                        {'params': classifier_nd,
                         'lr': self.base_lr * self.classifier_factor,
                         'weight_decay': 0.0},
                    ]
                    max_lrs += [
                        self.base_lr * self.classifier_factor,
                    ]

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

            self.loss_func = MTLLosses(tasks=self.tasks, weight=self.loss_factors)

            self.training_reducer = self.context.wrap_reducer(MTLPredictionsReducer(self.tasks, is_training=True),
                                                              for_training=True, for_validation=False)


        else:
            pass

    def build_training_data_loader(self) -> DataLoader:
        return DataLoader(self.train_datasets, batch_size=self.context.get_per_slot_batch_size(), shuffle=True,
                          num_workers=self.n_workers, pin_memory=True)

    def build_validation_data_loader(self) -> DataLoader:
        # not compatible for eval_full_datasets
        # self.eval_reducer = self.context.wrap_reducer(MTLPredictionsReducer(self.tasks, is_training=False),
        #                                               for_training=False, for_validation=True)

        return DataLoader(self.test_datasets, batch_size=self.context.get_per_slot_batch_size(), shuffle=False,
                          num_workers=self.n_workers, pin_memory=True)

    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, Any]:
        # self.mem_tracker.track()
        # print('========= after data to =========')
        # self.reporter.report(verbose=True)

        # if batch_idx == 0 or self.last_epoch_idx < epoch_idx:
        #     current_lr = self.lr_scheduler.get_last_lr()[0]
        #     print("Epoch: {} lr {}".format(epoch_idx, current_lr))
        # self.last_epcoch_idx = epoch_idx

        img = batch[0]
        seg_idx = self.tasks.index('seg')
        labels = {'task_{}'.format(t): batch[t + 1] if t != seg_idx else batch[t + 1].squeeze(1) for t in
                  range(self.n_tasks)}
        features = {'task_{}'.format(i): img for i in range(self.n_tasks)}

        out = self.model(features)
        loss = self.loss_func(out, labels)

        # print('========= before backward =========')
        # self.reporter.report(verbose=True)
        self.context.backward(loss['weighted_loss'])
        # print('========= after backward =========')
        # self.reporter.report(verbose=True)
        self.context.step_optimizer(self.optimizer)
        # print('========= after train =========')
        # self.reporter.report(verbose=True)

        # the lr scheduler step has already adopted by pytorch trial
        # see determined\pytorch\_pytorch_trial.py#Line404
        # self.lr_scheduler.step()

        self.training_reducer.update_train({k: loss['task_{}'.format(i)] for i, k in enumerate(self.tasks)})
        # print(torch.cuda.memory_summary())
        # print(torch.cuda.memory_stats_as_nested_dict())

        # the loss will be mean-reduced across the epoch
        return {k: v if k != 'weighted_loss' else v.item() for k, v in loss.items()}

    # fixme: may be there are some undefined behaviors for segmentation validation
    #        due to the distributed determined-ai reducers
    # def evaluate_batch(self, batch: TorchData, batch_idx: int) -> Dict[str, Any]:
    #     img = batch[0]
    #     seg_idx = self.tasks.index('seg')
    #     labels = {'task_{}'.format(t): batch[t + 1] if t != seg_idx else batch[t + 1].squeeze(1) for t in
    #               range(self.n_tasks)}
    #
    #     seg_pred, normal_pred = self.model(img, img)
    #     out = {'task_0': seg_pred, 'task_1': normal_pred}
    #     loss = self.loss_func(out, labels)
    #
    #     self.eval_reducer.update_eval(out, labels, {k: loss['task_{}'.format(i)] for i, k in enumerate(self.tasks)})
    #
    #     metrics = {k: v for k, v in loss.items() if k != 'weighted_loss'}
    #     metrics.update({'validation_loss': loss['weighted_loss'].item()})
    #
    #     return metrics

    def evaluate_full_dataset(self, data_loader: DataLoader) -> Dict[str, Any]:
        # only evaluate model with single slot
        _cnt = 0
        # fixme: currently only implemented for normal, segmentation
        if 'normal' in self.tasks:
            cosine = []  # type: List[np.array]
            cosine_mask = []  # type: List[np.array]
            _cnt += 1

        if 'seg' in self.tasks:
            conf_mat_sum = 0
            _cnt += 1

        assert self.n_tasks == _cnt

        n_batch = len(data_loader)
        batch_metrics = []  # type: List[Dict]
        for _, batch in enumerate(data_loader):

            # nddr_attention
            # img = self.context.to_device(batch[0])
            # seg_idx = self.tasks.index('seg')
            # labels = {
            #     'task_{}'.format(t): self.context.to_device(batch[t + 1]) if t != seg_idx
            #     else self.context.to_device(batch[t + 1]).squeeze(1) for t in range(self.n_tasks)}
            #
            # seg_pred, normal_pred = self.model(img, img)
            # out = {'task_0': seg_pred, 'task_1': normal_pred}

            # unites
            img = self.context.to_device(batch[0])
            seg_idx = self.tasks.index('seg')
            labels = {
                'task_{}'.format(t): self.context.to_device(batch[t + 1]) if t != seg_idx
                else self.context.to_device(batch[t + 1]).squeeze(1) for t in range(self.n_tasks)}
            features = {'task_{}'.format(i): img for i in range(self.n_tasks)}
            out = self.model(features)

            loss = self.loss_func(out, labels)

            # loss for each task has been applied with .detach().item()
            _metrics = {k: v for k, v in loss.items() if k != 'weighted_loss'}
            _metrics.update({'validation_loss': loss['weighted_loss'].item()})
            batch_metrics.append(_metrics)

            for i, t in enumerate(self.tasks):
                if t == 'seg':
                    # fixme: make the num of classes be configurable
                    _conf_mat = get_conf_mat(out['task_{}'.format(i)], labels['task_{}'.format(i)], 40)
                    conf_mat_sum += _conf_mat.cpu().numpy()
                elif t == 'normal':
                    _cosine, _mask = get_normal_cosine(prediction=out['task_{}'.format(i)],
                                                       gt=labels['task_{}'.format(i)], normalization=False,
                                                       ignore_label=255)
                    cosine.append(_cosine.cpu().numpy())
                    cosine_mask.append(_mask.cpu().numpy())
                else:
                    raise NotImplementedError('[EE] Unimplemented tasks: {}'.format(i))

        # statistics
        _reduced_metrics = {'task_losses': {}}

        _reduced_metrics.update({'validation_loss': sum([p['validation_loss'] for p in batch_metrics]) / n_batch})

        _cnt = 0
        if 'normal' in self.tasks:
            normal_idx = self.tasks.index('normal')
            _normal_loss = sum([p['task_{}'.format(normal_idx)] for p in batch_metrics]) / n_batch
            _reduced_metrics['task_losses'].update({'normal': _normal_loss})

            _cos_angels = [c[m] for c, m in zip(cosine, cosine_mask)]
            normal_metrics = get_normal_metrics(_cos_angels)

            _reduced_metrics.update({**normal_metrics})
            _cnt += 1

        if 'seg' in self.tasks:
            seg_idx = self.tasks.index('seg')
            _seg_loss = sum([p['task_{}'.format(seg_idx)] for p in batch_metrics]) / n_batch
            _reduced_metrics['task_losses'].update({'seg': _seg_loss})

            _mIoU = get_mIoU(conf_mat_sum) * 100.
            _pixel_acc = np.diag(conf_mat_sum).sum() / np.sum(conf_mat_sum) * 100.
            _reduced_metrics.update({'mIoU': _mIoU, 'pixel_acc': _pixel_acc})
            _cnt += 1

        assert self.n_tasks == _cnt

        return _reduced_metrics
