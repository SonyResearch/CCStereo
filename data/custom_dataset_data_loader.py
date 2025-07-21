#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np


def CreateDataset(opt, mode=''):
    dataset = None
    # if opt.model == 'audioVisual':
    from data.audioVisual_dataset import FAIRPLAY_DATSET
    dataset = FAIRPLAY_DATSET(mode)
    # else:
        # raise ValueError("Dataset [%s] not recognized." % opt.model)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

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

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, mode='train'):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size if mode=='train' else 1,
            shuffle=True if mode=='train' else False,
            num_workers=int(opt.num_workers),
            drop_last=True if mode=='train' else False,
            collate_fn=collation_fn
        )

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
