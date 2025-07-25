# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import optim as optim
from .Lion_opti import Lion
# from pytorch_lamb import Lamb

def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config['optimizer']['type'].lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config['optimizer']['momentum'], nesterov=True,
                              lr=config['base_lr'], weight_decay=config['weight_decay'])
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config['optimizer']['eps'], betas=config['optimizer']['betas'],
                                lr=config['base_lr'], weight_decay=config['weight_decay'])
    
    elif opt_lower == 'lion':
        # optimizer = optim.Lion(parameters, eps=config['optimizer']['eps'], betas=config['optimizer']['betas'],
        #                         lr=config['base_lr'], weight_decay=config['weight_decay'])
        optimizer = Lion(parameters, lr=config['base_lr'], weight_decay=config['weight_decay'],
                         betas=config['optimizer']['betas'])
    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
