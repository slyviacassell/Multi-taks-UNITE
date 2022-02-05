# coding=utf-8

'''
Created: 2021/3/1
@author: Slyviacassell@github.com
'''

# pytorch
import torch
import torch.nn as nn

# utils
from typing import (
    List,
    Dict
)

# costume
from .metrics import normal_loss


def get_losses(tasks_cfg: List) -> Dict:
    losses = dict()
    for i, t in enumerate(tasks_cfg):
        if t == 'normal':
            losses['task_{}'.format(i)] = normal_loss
        elif t == 'seg':
            losses['task_{}'.format(i)] = nn.CrossEntropyLoss(ignore_index=255)
        else:
            raise NotImplementedError('[EE] Not implemented loss for task: {}'.format(i))

    return losses


class MTLLosses(nn.Module):
    def __init__(self, tasks: List, weight: List):
        super(MTLLosses, self).__init__()
        self.tasks = tasks
        self.losses = get_losses(tasks)
        self.weight = weight

    def forward(self, predictions: Dict, label: Dict):
        loss = {}
        _weighted_loss = []
        for t, _ in enumerate(self.tasks):
            _loss = self.losses['task_{}'.format(t)](predictions['task_{}'.format(t)], label['task_{}'.format(t)])
            _weighted_loss.append(self.weight[t] * _loss)
            # for logging
            loss.update({'task_{}'.format(t): _loss.detach().cpu().item()})
        # loss.update({'weighted_loss': torch.sum(torch.stack(_weighted_loss))})
        loss.update({'weighted_loss': sum(_weighted_loss)})
        return loss
