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

from .UnGlobalAveragePool2d import UnGlobalAveragePool2d
from .GlobalAveragePool2d import GlobalAveragePool2d


class UNITELayer(nn.Module):
    def __init__(self, out_channels,
                 n_tasks,
                 n_patches,
                 reconstruct_src=None,
                 init_weights=[0.9, 0.1],
                 init_method='constant',
                 activation='relu',
                 batch_norm=True,
                 bn_before_activation=True,
                 conv_bias=False,
                 similarity='bilinear'):
        super(UNITELayer, self).__init__()

        assert isinstance(n_patches, int) or isinstance(n_patches, list)
        assert similarity in ['bilinear', 'additive', 'scaled-dot']
        if isinstance(n_patches, int):
            n_patches = [n_patches, n_patches]
        self.n_patches = n_patches
        self.n_tasks = n_tasks
        self.similarity = similarity
        self.reconstruct_src = reconstruct_src

        if similarity == 'additive':
            # additive
            assert reconstruct_src in ['raw', 'unpool']
            self.linears = nn.ModuleDict()
            self.vs = nn.ParameterDict()
            for i in range(n_tasks):
                for j in range(i + 1, n_tasks):
                    self.linears.update({'task_{}_{}'.format(i, j): nn.ModuleDict({
                        'linear_1': nn.Linear(out_channels, out_channels, bias=False),
                        'linear_2': nn.Linear(out_channels, out_channels, bias=False),
                    })})
                    self.vs.update({'task_{}_{}'.format(i, j):
                                        nn.Parameter(nn.init.normal_(torch.empty(out_channels)))})

            self.global_pool = GlobalAveragePool2d()
            self.unglobal_pool = UnGlobalAveragePool2d()
        elif similarity == 'scaled-dot':
            # scaled-dot product
            self.global_pool = GlobalAveragePool2d()
            self.unglobal_pool = UnGlobalAveragePool2d()
        elif similarity == 'bilinear':
            # bilinear
            self.register_parameter('bilinear_w', None)

        self.batch_norm = batch_norm
        self.bn_before_activation = bn_before_activation

        if batch_norm:
            self.bns = nn.ModuleDict()

        # Information Complementarity
        self.ics = nn.ModuleDict()
        for i in range(n_tasks):
            self.ics.update(
                {'task_' + str(i): nn.Conv2d(out_channels * n_tasks, out_channels, kernel_size=1, bias=conv_bias)})
                # ablation ccr
                # {'task_' + str(i): nn.Conv2d(out_channels * (n_tasks-1), out_channels, kernel_size=1, bias=conv_bias)})
            if self.batch_norm:
                self.bns.update({'task_' + str(i): nn.BatchNorm2d(out_channels, eps=1e-03, momentum=0.05)})

        assert len(init_weights) == 2 or len(init_weights) == n_tasks

        # init the ics
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

    def _pad(self, features: Dict):
        patches_sizes = {}
        pad_sizes = {}
        padded_features = {}
        for k, v in features.items():
            _size = v.size()
            _patch_size = [_size[2] // self.n_patches[0], _size[3] // self.n_patches[1]]
            _pad = (_size[3] % _patch_size[1] // 2, _size[3] % _patch_size[1] // 2,
                    _size[2] % _patch_size[0] // 2, _size[2] % _patch_size[0] // 2)
            patches_sizes.update({k: _patch_size})
            pad_sizes.update({k: _pad})
            padded_features.update({k: F.pad(v, _pad)})

        return padded_features, patches_sizes, pad_sizes

    def _reconstruct_with_sim(self, features: Dict, sim: Dict) -> Dict:
        _recon_feas = {'task_{}'.format(i): {} for i in range(self.n_tasks)}
        for i in range(self.n_tasks):
            for j in range(i + 1, self.n_tasks):
                if self.reconstruct_src == 'raw' or self.similarity == 'scaled-dot':
                    _rf_2 = torch.einsum('bij,bjk->bik', features['task_{}'.format(i)],
                                         F.softmax(sim['task_{}_{}'.format(i, j)], dim=1))
                    _rf_1 = torch.einsum('bij,bjk->bik', features['task_{}'.format(j)],
                                         F.softmax(sim['task_{}_{}'.format(i, j)].transpose(1, 2), dim=1))
                else:
                    _rf_2 = torch.einsum('bij,bjk->bik', features['task_{}'.format(i)].transpose(1, 2),
                                         F.softmax(sim['task_{}_{}'.format(i, j)], dim=1))
                    _rf_1 = torch.einsum('bij,bjk->bik', features['task_{}'.format(j)].transpose(1, 2),
                                         F.softmax(sim['task_{}_{}'.format(i, j)].transpose(1, 2), dim=1))

                _recon_feas['task_{}'.format(j)].update({'task_{}'.format(i): _rf_1})
                _recon_feas['task_{}'.format(i)].update({'task_{}'.format(j): _rf_2})

        return _recon_feas

    def _scaled_dot(self, features: Dict, patch_sizes: Dict):
        sim = {}
        padded_sizes = {}

        for k,v in features.items():
            _padded_size = v.size()
            padded_sizes.update({k: _padded_size})
            features.update({k:F.unfold(v, kernel_size=patch_sizes[k], stride=patch_sizes[k])})

        # similarity
        for i in range(self.n_tasks):
            for j in range(i + 1, self.n_tasks):
                sim.update({'task_{}_{}'.format(i, j):
                                torch.einsum('bij,bjk->bik',
                                             features['task_{}'.format(i)].transpose(1, 2),
                                             features['task_{}'.format(j)]) /
                                torch.sqrt(torch.tensor(features['task_{}'.format(i)].size(1), dtype=torch.float))})

        features = self._reconstruct_with_sim(features, sim)

        for k, v in features.items():
            _tmp = []
            for _k, _v in v.items():
                _rf = F.fold(_v, output_size=padded_sizes[k][2:], kernel_size=patch_sizes[k], stride=patch_sizes[k])
                _tmp.append(_rf)
            features.update({k: torch.cat(_tmp, dim=1)})

        return features

    def _additive(self, features: Dict, patch_sizes: Dict):
        padded_sizes = {}
        unfolded_sizes = {}
        new_heights = {}
        new_widths = {}
        sim = {}

        # used for raw
        patches_features = {}

        for k, v in features.items():
            _padded_size = v.size()
            _tmp = v.unfold(2, patch_sizes[k][0], patch_sizes[k][0]) \
                    .unfold(3, patch_sizes[k][1], patch_sizes[k][1]).contiguous()
            _unfolded_size = _tmp.size()

            if self.reconstruct_src == 'raw':
                _patches = _tmp.view(*_unfolded_size[:-2], -1) \
                    .permute(0, 1, 4, 2, 3) \
                    .contiguous() \
                    .view(_unfolded_size[0], -1, _unfolded_size[2] * _unfolded_size[3])
                patches_features.update({k: _patches})

            _tmp = self.global_pool(_tmp)
            _tmp = _tmp.view(_tmp.size(0), _tmp.size(1), -1)\
                    .transpose(1, 2)

            unfolded_sizes.update({k: _unfolded_size})
            new_heights.update({k: _unfolded_size[-4]})
            new_widths.update({k: _unfolded_size[-3]})
            padded_sizes.update({k: _padded_size})
            features.update({k: _tmp})

        # similarity
        for i in range(self.n_tasks):
            for j in range(i + 1, self.n_tasks):
                sim.update({'task_{}_{}'.format(i, j): torch.einsum('bijk,k->bij', F.tanh(
                    self.linears['task_{}_{}'.format(i, j)]['linear_1'](features['task_{}'.format(i)]).unsqueeze_(-2) +
                    self.linears['task_{}_{}'.format(i, j)]['linear_2'](features['task_{}'.format(j)]).unsqueeze_(-3)),
                                                                    self.vs['task_{}_{}'.format(i, j)].t())})

        if self.reconstruct_src == 'raw':
            # use raw patches to reconstruct
            features = self._reconstruct_with_sim(patches_features, sim)

            recon_sum={}

            for k, v in features.items():
                _tmp = []
                for _k, _v in v.items():
                    _rf = F.fold(_v, padded_sizes[k][2:], kernel_size=patch_sizes[k], stride=patch_sizes[k])
                    _tmp.append(_rf)
                features.update({k: torch.cat(_tmp, dim=1)})

                recon_sum.update({k:sum(_tmp)})

        elif self.reconstruct_src == 'unpool':
            features = self._reconstruct_with_sim(features, sim)

            recon_sum={}

            for k, v in features.items():
                _tmp = []
                for _k, _v in v.items():
                    _unpooled = _v.view(*_v.size()[:-1], new_heights[k], new_widths[k])
                    _unpooled = self.unglobal_pool(_unpooled, size=patch_sizes[k])
                    _unpooled = _unpooled.view(*unfolded_sizes[k][:-2], -1) \
                        .permute(0, 1, 4, 2, 3) \
                        .contiguous() \
                        .view(unfolded_sizes[k][0], unfolded_sizes[k][1] * patch_sizes[k][0] * patch_sizes[k][1],
                              unfolded_sizes[k][2] * unfolded_sizes[k][3])
                    _unpooled = F.fold(_unpooled, padded_sizes[k][-2:], kernel_size=patch_sizes[k],
                                       stride=patch_sizes[k])
                    _tmp.append(_unpooled)
                features.update({k: torch.cat(_tmp, dim=1)})


        else:
            raise ValueError('[EE] Unknown reconstruct method: {}'.format(self.reconstruct_src))

        return features

    def _bilinear(self, features: Dict):
        pass

    def forward(self, features: Dict) -> Dict:
        feature_sizes = {k: v.size() for k, v in features.items()}

        _features, patches_sizes, pad_sizes = self._pad(features)

        if self.similarity == 'additive':
            # _recon_feas, recon_sum = self._additive(_features, patches_sizes)
            _recon_feas = self._additive(_features, patches_sizes)
        elif self.similarity == 'scaled-dot':
            _recon_feas = self._scaled_dot(_features, patches_sizes)
        else:
            raise NotImplementedError('[EE] Not Implemented similarity: {}'.format(self.similarity))

        _loss=0.

        for k, v in _recon_feas.items():

            _cated_fea = torch.cat([features[k], v[:, :, pad_sizes[k][2]:pad_sizes[k][2] + feature_sizes[k][2],
                                                 pad_sizes[k][0]:pad_sizes[k][0] + feature_sizes[k][3]]], dim=1)

            # ablation ccr
            # _cated_fea = v[:, :, pad_sizes[k][2]:pad_sizes[k][2] + feature_sizes[k][2],
            #                                      pad_sizes[k][0]:pad_sizes[k][0] + feature_sizes[k][3]]

            _out_fea = self.ics[k](_cated_fea)
            if self.batch_norm and self.bn_before_activation:
                _out_fea = self.bns[k](_out_fea)
            if self.activation:
                _out_fea = self.activation(_out_fea)
            if self.batch_norm and not self.bn_before_activation:
                _out_fea = self.bns[k](_out_fea)

            features.update({k: _out_fea})

        return features
