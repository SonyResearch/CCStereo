#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from data.fairplay_dataset import FAIRPLAY_DATSET
from data.yt_clean_dataset import YT_CLEAN_DATSET


def collation_fn(samples):
    batched = list(zip(*samples))
    result = []
    for b in batched:
        if isinstance(b[0], (int, float)):
            b = np.array(b)
        elif isinstance(b[0], torch.Tensor):
            b = torch.stack(b)
        elif isinstance(b[0], np.ndarray):
            b = np.array(b)
        else:
            b = b
        result.append(b)
    return result

def make_fairplay_loader(opt, mode='train'):
    batch_size = opt.batch_size if mode=='train' else 1
    shuffle = True if mode=='train' else False
    drop_last = True if mode=='train' else False
    logger.info(f"Creating {mode} dataloader with batch size {batch_size}, shuffle {shuffle}, drop_last {drop_last}")
    dataset = FAIRPLAY_DATSET(opt, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(opt.num_workers),
        drop_last=drop_last,
        collate_fn=collation_fn
    )
    return dataloader

def make_yt_clean_loader(opt, mode='train'):
    batch_size = opt.batch_size if mode=='train' else 1
    shuffle = True if mode=='train' else False
    drop_last = True if mode=='train' else False
    logger.info(f"Creating {mode} dataloader with batch size {batch_size}, shuffle {shuffle}, drop_last {drop_last}")
    dataset = YT_CLEAN_DATSET(opt, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(opt.num_workers),
        drop_last=drop_last,
        collate_fn=collation_fn
    )
    return dataloader

def CreateDataLoader(opt):
    if opt.dataset == 'fairplay':
        train_loader = make_fairplay_loader(opt, 'train')
        val_loader = make_fairplay_loader(opt, 'val')
        test_loader = make_fairplay_loader(opt, 'test')
    elif opt.dataset == 'yt_clean':
        train_loader = make_yt_clean_loader(opt, 'train')
        val_loader = make_yt_clean_loader(opt, 'test')
        test_loader = make_yt_clean_loader(opt, 'test')
    else:
        raise ValueError(f"Unknown dataset {opt.dataset}")
    
    return train_loader, val_loader, test_loader