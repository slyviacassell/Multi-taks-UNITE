# coding=utf-8

'''
Created: 2021/3/1
@author: Slyviacassell@github.com
'''

# pytorch
import torch

# determined
from determined.pytorch import MetricReducer

# utils
import numpy as np
from typing import (
    Any,
    Dict
)

# costume
from .metrics import get_mIoU, get_normal_metrics, get_normal_cosine, get_conf_mat


class MTLPredictionsReducer(MetricReducer):
    '''Only used for a multi-slot trail'''

    def __init__(self, tasks, is_training=True):
        super(MTLPredictionsReducer, self).__init__()
        self.tasks = tasks
        self.n_task = len(tasks)
        self.is_training = is_training

        # batch-reduced loss
        self.loss = {i: 0. for i in self.tasks}
        self.batch_loss_cnt = 0

        if not self.is_training:
            self._init_eval_statistics()

    def _init_eval_statistics(self):

        _cnt = 0
        if 'normal' in self.tasks:
            # np.array
            self.cosine = []
            self.cosine_mask = []
            _cnt += 1

        if 'seg' in self.tasks:
            self.conf_mat = 0
            _cnt += 1

        assert self.n_task == _cnt

    def per_slot_reduce(self) -> Any:
        _slot_metrics = {'losses': self.loss}
        _cnt = 0

        if 'normal' in self.tasks and not self.is_training:
            _slot_metrics.update({'normal_angles': np.concatenate(self.cosine, axis=0),
                                  'angels_mask': np.concatenate(self.cosine_mask, axis=0)})

            _cnt += 1

        if 'seg' in self.tasks and not self.is_training:
            _slot_metrics.update({'seg_conf_mat': self.conf_mat})

            _cnt += 1

        if not self.is_training:
            assert self.n_task == _cnt

        return _slot_metrics

    def cross_slot_reduce(self, per_slot_metrics) -> Any:
        _reduced_metrics = {'epoch_losses': {}}

        # average_training_metrics == false
        world_size = len(per_slot_metrics)

        _cnt = 0
        if 'normal' in self.tasks:
            _normal_loss = sum([p['losses']['normal'] for p in per_slot_metrics]) / (self.batch_loss_cnt * world_size)
            _reduced_metrics['epoch_losses'].update({'normal': _normal_loss})
            if not self.is_training:
                _cos_angels = [p['normal_angles'][p['angels_mask']] for _, p in enumerate(per_slot_metrics)]
                normal_metrics = get_normal_metrics(_cos_angels)

                _reduced_metrics.update({**normal_metrics})
                _cnt += 1

        if 'seg' in self.tasks:
            _seg_loss = sum([p['losses']['seg'] for p in per_slot_metrics]) / (self.batch_loss_cnt * world_size)
            _reduced_metrics['epoch_losses'].update({'seg': _seg_loss})
            if not self.is_training:
                _conf_mat_sum = sum([p['seg_conf_mat'] for p in per_slot_metrics])
                _mIoU = get_mIoU(_conf_mat_sum)
                _pixel_acc = np.diag(_conf_mat_sum).sum() / np.sum(_conf_mat_sum)
                _reduced_metrics.update({'mIoU': _mIoU, 'pixel_acc': _pixel_acc})
                _cnt += 1

        if not self.is_training:
            assert self.n_task == _cnt

        return _reduced_metrics

    def reset(self) -> None:
        self.loss.update({i: 0. for i in self.tasks})
        self.batch_loss_cnt = 0
        if self.is_training:
            self._init_eval_statistics()

    # for eval_batch
    def update_eval(self, prediction: Dict[str, torch.Tensor], label: Dict[str, torch.Tensor], losses: Dict, cnt=1):
        # all the tensor are detached
        self.update_train(losses, cnt)

        for i, t in enumerate(self.tasks):
            if t == 'seg':
                # fixme: make the num of classes be configurable
                _conf_mat = get_conf_mat(prediction['task_{}'.format(i)], label['task_{}'.format(i)], 40)
                self.conf_mat += _conf_mat.cpu().numpy()
            elif t == 'normal':
                _cosine, _mask = get_normal_cosine(prediction=prediction['task_{}'.format(i)],
                                                   gt=label['task_{}'.format(i)], normalization=False,
                                                   ignore_label=255)
                self.cosine.append(_cosine.cpu().numpy())
                self.cosine_mask.append(_mask.cpu().numpy())
            elif t == 'depth':
                pass
            else:
                raise NotImplementedError('[EE] Unimplemented tasks: {}'.format(i))

    def update_train(self, losses: Dict, cnt=1):
        # the loss must be detached from the computational graph
        self.batch_loss_cnt += cnt
        for k, v in losses.items():
            if k in self.tasks:
                self.loss.update({k: self.loss[k] + v})
            else:
                raise ValueError('[EE] Unknown task loss: {}'.format(k))
