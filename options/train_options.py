#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_options import BaseOptions
from easydict import EasyDict
import json
import torch
import random, os, numpy

class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--display_freq', type=int, default=50, help='frequency of displaying average loss')
		self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
		self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
		self.parser.add_argument('--epochs', type=int, default=100, help='# of epochs to train')
		self.parser.add_argument('--learning_rate_decrease_itr', type=int, default=-1, help='how often is the learning rate decreased by six percent')
		self.parser.add_argument('--decay_factor', type=float, default=0.94, help='learning rate decay factor')
		self.parser.add_argument('--tensorboard', type=bool, default=False, help='use tensorboard to visualize loss change ')		
		self.parser.add_argument('--measure_time', type=bool, default=False, help='measure time of different steps during training')
		self.parser.add_argument('--validation_on', action='store_true', help='whether to test on validation set during training')
		self.parser.add_argument('--validation_freq', type=int, default=100, help='frequency of testing on validation set')
		self.parser.add_argument('--validation_batches', type=int, default=10, help='number of batches to test for validation')
		self.parser.add_argument('--enable_data_augmentation', type=bool, default=True, help='whether to augment input frame')
		
		# data
		self.parser.add_argument('--data_vol_scaler', type=int, default=1, help="scaling factor for the volume of the training data")
		
		# model arguments
		self.parser.add_argument('--backbone', type=int, default=50)
		self.parser.add_argument('--method_type', type=str, default='m2b')
		self.parser.add_argument('--weights_visual', type=str, default='', help="weights for visual stream")
		self.parser.add_argument('--weights_audio', type=str, default='', help="weights for audio stream")
		self.parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
		self.parser.add_argument('--unet_input_nc', type=int, default=2, help="input spectrogram number of channels")
		self.parser.add_argument('--unet_output_nc', type=int, default=2, help="output spectrogram number of channels")
		self.parser.add_argument('--dim_scale', type=int, default=4)

		# optimizer arguments
		self.parser.add_argument('--optimizer', default='adam', type=str, help='adam or sgd for optimization')
		self.parser.add_argument('--weight_decay', default=0.0005, type=float, help='weights regularizer')
		self.parser.add_argument('--lr_img', type=float, default=5e-5, help='learning rate')
		self.parser.add_argument('--lr_aud', type=float, default=5e-4, help='learning rate')

		# self.mode = "train"
		# self.isTrain = True
		self.enable_data_augmentation = True

		self.parser.add_argument('--name', default="stable_audio_tools", type=str, help='name of the run')
		self.parser.add_argument('--batch_size', default=16, type=int, help='the batch size')
		self.parser.add_argument('--num_gpus', default=1, type=int, help='number of GPUs to use for training')
		self.parser.add_argument('--num_nodes', default=1, type=int, help='number of nodes to use for training')
		self.parser.add_argument('--strategy', default='', type=str, help='Multi-GPU strategy for PyTorch Lightning')
		self.parser.add_argument('--precision', default='16-mixed', type=str, help='Precision to use for training')
		self.parser.add_argument('--num_workers', default=16, type=int, help='number of CPU workers for the DataLoader')
		self.parser.add_argument('--seed', default=42, type=int, help='the random seed')
		self.parser.add_argument('--accum_batches', default=1, type=int, help='Batches for gradient accumulation')
		self.parser.add_argument('--checkpoint_every', default=10000, type=int, help='Number of steps between checkpoints')
		self.parser.add_argument('--ckpt_path', default='', type=str, help='trainer checkpoint file to restart training from')
		self.parser.add_argument('--pretrained_ckpt_path', default='', type=str, help='model checkpoint file to start a new training run from')
		self.parser.add_argument('--pretransform_ckpt_path', default='', type=str, help='Checkpoint path for the pretransform model if needed')
		self.parser.add_argument('--model_config', default='', type=str, help='configuration model specifying model hyperparameters')
		self.parser.add_argument('--dataset_config', default='', type=str, help='configuration for datasets')
		self.parser.add_argument('--save_dir', default='', type=str, help='directory to save the checkpoints in')
		self.parser.add_argument('--gradient_clip_val', default=0.0, type=float, help='gradient_clip_val passed into PyTorch Lightning Trainer')
		self.parser.add_argument('--remove_pretransform_weight_norm', default='', type=str, help='remove the weight norm from the pretransform model')
		#
		# wandb
		self.parser.add_argument("--wandb_mode", default="disabled", type=str, help="Mode of wandb API")
		self.parser.add_argument("--wandb_name", default="dummpy", type=str, help="Mode of wandb API")
		self.parser.add_argument("--wandb_dir", default="logs_ct", type=str, help="Mode of wandb API")
		self.parser.add_argument("--wandb_proj", default="stable_audio", type=str, help="Mode of wandb API")
		self.parser.add_argument("--run_tags", nargs="+", default="")
		self.parser.add_argument("--run_note", default="", type=str, help="Notes for run")
		self.parser.add_argument("--run_name", default="", type=str, help="Mode of wandb API")

		# dataset
		self.parser.add_argument('--data_root', default='./dataset', type=str)
		## augmentation
		self.parser.add_argument("--random_mono", default=False, action="store_true")

		# TRAINING
		self.parser.add_argument("--use_visual", default=False, action="store_true")
		self.parser.add_argument("--multi_frames", default=False, action="store_true")
		self.parser.add_argument("--use_flow", default=False, action="store_true")
		self.parser.add_argument("--use_mask", default=False, action="store_true")

		# TESTING
		self.parser.add_argument('--test_steps', default=50, type=int, help='Number of steps to test')
		self.parser.add_argument('--test_cfg_scale', default=6, type=int, help='Scale of the test config')
		self.parser.add_argument('--test_hop_size', default=0.1, type=float, help='Hop size of the test config')

		self.parser.add_argument('--ckpt-path', type=str, default="")
		self.parser.add_argument('--ckpt-id', type=str, default="")
		self.parser.add_argument('--use-safetensors', action='store_true')
		self.parser.add_argument('--save_preds', action='store_true')
		
	def parse(self,):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		self.opt = EasyDict(vars(self.opt))

		# self.opt.mode = self.mode
		# self.opt.isTrain = self.isTrain
		self.opt.enable_data_augmentation = self.enable_data_augmentation

		str_ids = self.opt.gpu_ids.split(',')
		self.opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu_ids.append(id)

		# set gpu ids
		if len(self.opt.gpu_ids) > 0:
			torch.cuda.set_device(self.opt.gpu_ids[0])


		#I should process the opt here, like gpu ids, etc.
		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')

		self.update_seed(self.opt.seed)

		if self.opt.dataset == 'fairplay':
			self.opt.data_root = './dataset/fairplay'
		elif self.opt.dataset == 'yt_clean':
			self.opt.data_root = './dataset/yt_clean'
		else:	
			raise ValueError(f"Dataset {self.opt.dataset} not supported")

		return self.opt#, self.model_config


	def update_seed(self, seed):
		random.seed(seed)
		os.environ["PYTHONSEED"] = str(seed)
		numpy.random.seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		# torch.backends.cudnn.deterministic = True
		# torch.backends.cudnn.benchmark = False
		# torch.backends.cudnn.enabled = True
		torch.manual_seed(seed)