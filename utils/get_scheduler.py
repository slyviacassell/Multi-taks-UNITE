# coding=utf-8

'''
Created: 2021/3/8
@author: Slyviacassell@github.com
'''

import torch.optim as optim

from typing import Dict


def get_lr_scheduler(optimizer, cfg: Dict):
    eps = 1e-12
    if cfg['scheduler'] == 'Poly':
        if cfg['n_warmup_units'] > 0:
            # add eps
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lambda step: min(1., float(step) / cfg['n_warmup_units'] + eps) * (
                                                            1 - float(step) / cfg['n_train_units']) ** cfg[
                                                                     'poly_power'] + eps,
                                                    # 1 - float(step) / cfg.TRAIN.EPOCHS) ** cfg.TRAIN.POWER,
                                                    last_epoch=-1)
        else:
            scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                    lambda step: (1 - float(step) / cfg['n_train_units']) \
                                                                 ** cfg['poly_power'] + eps,
                                                    # lambda step: (1 - float(step) / cfg.TRAIN.EPOCHS) ** cfg.TRAIN.POWER,
                                                    last_epoch=-1)
    elif cfg['scheduler'] == 'Cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['n_train_units'])
    elif cfg['scheduler'] == 'Step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, int(cfg['n_train_units'] * 0.25))
    elif cfg['scheduler'] == 'OneCycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg['base_lr'], total_steps=cfg['n_train_units'], )
    elif cfg['scheduler'] == 'LR-Finder-Exp':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                lambda step: (0.1 / cfg['base_lr']) ** (step / cfg['n_train_units']),
                                                last_epoch=-1)
    else:
        raise NotImplementedError('[EE] Unimplemented scheduler: {}'.format(cfg['scheduler']))

    return scheduler
