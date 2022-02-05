# coding=utf-8

'''
Created: 2021/3/1
@author: Slyviacassell@github.com
'''

import torch.nn as nn
import torch
import torch.nn.functional as F

from typing import (
    Dict,
    List,
    Union
)


class DoubleLayer(nn.Module):
    def __init__(self,
                 out_channels,
                 n_tasks,
                 init_method='constant',
                 init_weights=[0.9, 0.1],
                 activation='relu',
                 batch_norm=True,
                 bn_before_activation=True,
                 conv_bias=False, ):
        super(DoubleLayer, self).__init__()
        self.n_tasks = n_tasks
        self.batch_norm = batch_norm
        self.bn_before_activation = bn_before_activation
        assert len(init_weights) == 2 or len(init_weights) == n_tasks

        if batch_norm:
            self.bns = nn.ModuleDict()

        # Double Conv
        self.ics = nn.ModuleDict()
        for i in range(n_tasks):
            self.ics.update(
                {'task_' + str(i): nn.Conv2d(out_channels * n_tasks, out_channels, kernel_size=1, bias=conv_bias)})
            if self.batch_norm:
                self.bns.update({'task_' + str(i): nn.BatchNorm2d(out_channels, eps=1e-03, momentum=0.05)})

        # init
        if init_method == 'constant':
            if len(init_weights) == 2:
                # assume that the first init value is used to initialize the main task for a given information flow
                _auxiliary_init_weight = init_weights[1] / (n_tasks - 1)
                for i in range(n_tasks):
                    self.ics['task_' + str(i)].weight = nn.Parameter(
                        torch.cat(
                            [torch.eye(out_channels) * init_weights[0]] +
                            [torch.eye(out_channels) * _auxiliary_init_weight for i in range(n_tasks - 1)]
                            , dim=1
                        ).view(out_channels, -1, 1, 1))
                    if self.ics['task_' + str(i)].bias:
                        self.ics['task_' + str(i)].bias.data.fill_(0)
            else:
                # assume that the order of init value is along with that of task
                # i.e. ([t1, t2, t3], [v1, v2, v3]), ([t2, t1, t3], [v2, v1, v3])
                # maybe there will be some other implementations?
                _perm_list = [i for i in range(n_tasks)]
                for i in range(n_tasks):
                    self.ics['task_' + str(i)].weight = nn.Parameter(torch.cat([
                        torch.eye(out_channels) * init_weights[j] for j in _perm_list
                    ], dim=1).view(out_channels, -1, 1, 1))
                    if self.ics['task_' + str(i)].bias:
                        self.ics['task_' + str(i)].bias.data.fill_(0)

                    # permute the first init value
                    if i < n_tasks - 1:
                        _perm_list[0], _perm_list[i + 1] = _perm_list[i + 1], _perm_list[0]

        elif init_method == 'xavier':
            for i in range(n_tasks):
                nn.init.xavier_uniform_(self.ics['task_' + str(i)].weight)
                if self.ics['task_' + str(i)].bias:
                    self.ics['task_' + str(i)].bias.data.fill_(0)

        elif init_method == 'kaiming':
            for i in range(n_tasks):
                nn.init.kaiming_normal_(self.ics['task_' + str(i)].weight, mode='fan_out', nonlinearity='relu')
                if self.ics['task_' + str(i)].bias:
                    self.ics['task_' + str(i)].bias.data.fill_(0)
        else:
            raise NotImplementedError('[EE] Init method {} is not implemented!'.format(init_method))

        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError('[EE] Activation {} is not implemented!'.format(activation))

    def forward(self, features: Dict) -> Dict:
        for k, v in features.items():
            _cated_fea = torch.cat([v, v], dim=1)
            _out_fea = self.ics[k](_cated_fea)
            if self.batch_norm and self.bn_before_activation:
                _out_fea = self.bns[k](_out_fea)
            if self.activation:
                _out_fea = self.activation(_out_fea)
            if self.batch_norm and not self.bn_before_activation:
                _out_fea = self.bns[k](_out_fea)

            features.update({k: _out_fea})

        return features
