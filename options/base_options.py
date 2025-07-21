#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import torch
from easydict import EasyDict

def add_tag(tags, key):
    if len(tags) != 0:
        tags.append(key)
    else:
        tags = [key]
    return tags

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--setup', type=str, choices=['5splits', '10splits'], default="10splits")
		self.parser.add_argument('--hdf5FolderPath', default="split1", help='path to the folder that contains train.h5, val.h5 and test.h5')
		self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		# self.parser.add_argument('--name', type=str, default='stable_audio', help='name of the experiment. It decides where to store models')
		# self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/', help='models are saved here')
		# self.parser.add_argument('--model', type=str, default='audioVisual', help='chooses how datasets are loaded.')
		# self.parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
		# self.parser.add_argument('--nThreads', default=16, type=int, help='# threads for loading data')
		self.parser.add_argument('--audio_sampling_rate', default=16000, type=int, help='audio sampling rate')
		self.parser.add_argument('--audio_length', default=1.0, type=float, help='audio length, default 0.63s')
        #
		self.parser.add_argument('--dataset', default="fairplay", choices=["fairplay", "yt_clean"], type=str)
		self.enable_data_augmentation = True
		self.initialized = True

	