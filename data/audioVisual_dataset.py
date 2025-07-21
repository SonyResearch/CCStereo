#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os.path
import time
import librosa
import h5py
import random
import math
import numpy as np
from glob import glob
import torch
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
import torch.nn.functional as F
from scipy.signal import resample
import torch.nn as nn
from torch.utils.data import Dataset
from loguru import logger

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = torch.maximum(torch.tensor(eps), torch.sqrt(torch.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return rms / desired_rms, samples

def generate_spectrogram(audio):
    spectro = librosa.core.stft(audio, n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel

def process_image(image, augment):
    image = image.resize((480,240))
    w,h = image.size
    w_offset = w - 448
    h_offset = h - 224
    left = random.randrange(0, w_offset + 1)
    upper = random.randrange(0, h_offset + 1)
    image = image.crop((left, upper, left+448, upper+224))

    if augment:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image

class FAIRPLAY_DATSET(Dataset):
    def __init__(self, opt, mode=''):
        super().__init__()
        self.initialize(opt, mode)

    def initialize(self, opt, mode=''):
        self.opt = opt
        self.audios = []
        self.mode = mode

        #load hdf5 file here
        h5f_path = os.path.join(opt.data_root, opt.hdf5FolderPath, mode+".h5")
        
        logger.info(f"Loading {mode} data from <<< {h5f_path} >>>")
        h5f = h5py.File(h5f_path, 'r')
        self.audios = h5f['audio'][:]

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)
        self.sample_rate = 16000
        self.hop_length = 512
        self.random_mono = opt.random_mono
        self.data_root = opt.data_root
        # 
        self.sample_size = opt.sample_size # 16384
        self.test_hop_size = opt.test_hop_size # 0.1
        # self.samples_per_window = opt.sample_size # 16384
        self.samples_per_window = int(opt.audio_length * self.sample_rate)

        logger.info(
            f"Test hop size: {self.test_hop_size}, Samples per window: {self.samples_per_window}"
        )

        from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Reverse

        self.audio_augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            # Reverse(p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(p=0.5),
        ])

        # from torch_audiomentations import Compose, Gain, PolarityInversion, PitchShift
        # self.audio_augment= Compose(
        #     transforms=[
        #         Gain(
        #             min_gain_in_db=-15.0,
        #             max_gain_in_db=5.0,
        #             p=0.5,
        #         ),
        #         # PitchShift(min_transpose_semitones=-4, max_transpose_semitones=4, p=0.5),
        #         PolarityInversion(p=0.5),
        #     ]
        # )

    def process_audio_augment(self, audio):
        audio = audio.copy().astype(np.float32)
        # sample_rate = self.sample_rate + torch.randint(-100, 100, (1,)).item()
        # audio = self.audio_augment(audio, sample_rate=self.sample_rate)
        audio = torch.tensor(audio)
        if random.random() < 0.5:
            audio = torch.flip(audio, dims=[0])
        return audio
    
    def compute_padding(self, length, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate
        right_pad = math.ceil(length / self.sample_size) * self.sample_size - length
        return right_pad
    
    def generate_audio_clips(self, input_audio):
        # input: (c, sample_rate * time)
        input_size_ = input_audio.shape[-1]
        hop_size_ = int(self.test_hop_size * self.samples_per_window)
        slide_num = int(np.ceil((input_size_ - self.samples_per_window) / hop_size_))
        # config the window start positions
        windows_list = range(0, (slide_num + 1) * hop_size_, hop_size_)
        # compute audio len after padding
        final_len = slide_num * hop_size_ + self.samples_per_window
        audio_clips = F.pad(input_audio, (0, final_len - input_size_))
        audio_clips = audio_clips.unfold(dimension=-1, size=self.samples_per_window, step=hop_size_) # (c, num_windows, samples_per_window)
        audio_clips = audio_clips.permute(1,0,2)# (num_windows, c, samples_per_window)
        return audio_clips, windows_list
    
    def prepare_audio(self, file_path):
        audio_file = os.path.join(self.data_root, "binaural_audios", file_path.split("/")[-1])
        audio, audio_rate = librosa.load(audio_file, sr=self.sample_rate, mono=False)
        if self.mode == 'train':
            # randomly get a start time for the audio segment from the 10s clip
            audio_start_time = random.uniform(0, 9.9 - self.opt.audio_length)
            audio_end_time = audio_start_time + self.opt.audio_length
            audio_start = int(audio_start_time * self.sample_rate)
            audio_end = audio_start + int(self.opt.audio_length * self.sample_rate)
            
            # pad_ = self.compute_padding((audio_end-audio_start), self.sample_rate)
            # audio = audio[:, audio_start:(audio_end+pad_)]
            
            audio = audio[:, audio_start:audio_end]
            # augment the audio
            audio = self.process_audio_augment(audio)
            # audio = torch.tensor(audio)
            return audio, (audio_start_time, audio_end_time), torch.empty(0), None
        else:
            audio = torch.tensor(audio)
            # unfold the audio into clips
            audio_clips, windows_list = self.generate_audio_clips(audio)
            return audio, None, audio_clips, windows_list
    
    def prepare_diff_audio(self, audio):
        audio_stereo = audio.clone()
        audio_tmp = audio.clone()

        # following 2.5D's benchmark
        normalizer, audio_tmp = normalize(audio_tmp)

        audio_left = audio_tmp[0, None] # L
        audio_right = audio_tmp[1, None] # R

        audio_diff = audio_left - audio_right
        audio_mono = audio_left + audio_right

        return audio_stereo, audio_mono, audio_diff

    def prepare_visual(self, file_path, clip_info=None, audio_len=None, windows_list=None, audio_num_samples=None):
        path_parts = file_path.strip().split('/')
        if self.mode == 'train':
            audio_start_time, audio_end_time = clip_info
            path_parts = file_path.strip().split('/')
            frame_path = os.path.join(self.data_root, 'frames', path_parts[-1][:-4] + '.mp4')
            # get the closest frame to the audio segment
            # frame_index = int(round((audio_start_time + audio_end_time) / 2.0 + 0.5))  #1 frame extracted per second
            frame_index = int(round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 10))  #10 frames extracted per second
            frames = process_image(Image.open(os.path.join(frame_path, str(frame_index).zfill(6) + '.png')).convert('RGB'), True)
            frames = self.vision_transform(frames)
        else:
            frame_path = os.path.join(self.data_root, 'frames', path_parts[-1][:-4] + '.mp4')
            num_frames = len(glob(os.path.join(frame_path, '*.png')))
            # frame_idx_list = [max(0, int((i/self.samples_per_window)*(num_frames/audio_len))-1) for i in windows_list]
            # ((i + self.samples_per_window / 2.0) / audio_num_samples) -> current audio second in terms of total
            frame_idx_list = [max(0, int(round((((i + self.samples_per_window / 2.0) / audio_num_samples) * audio_len + 0.05) * 10))) for i in windows_list]
            frames = [] 
            for i in frame_idx_list:
                if i > num_frames:
                    assert i < num_frames, f"Frame index {i} is greater than the number of frames {num_frames}"
                frame = process_image(Image.open(os.path.join(frame_path, str(i).zfill(6) + '.png')).convert('RGB'), False)
                frame = self.vision_transform(frame)
                frames.append(frame)
            frames = torch.stack(frames)
        return frames
    
    def load_batch(self, index):
        file_path = self.audios[index].decode('UTF-8')
        audio, clip_info, audio_clips, windows_list = self.prepare_audio(file_path)
        audio_stereo, audio_mono, audio_diff = self.prepare_diff_audio(audio)
        #
        kwgs = {
            "clip_info": clip_info,
            "audio_len": int(audio.shape[-1]/self.sample_rate),
            "windows_list": windows_list,
            "audio_num_samples": audio.shape[-1]
        }
        frames = self.prepare_visual(file_path, **kwgs)
        metadata = {
            "seconds_start": int(0.0), 
            "seconds_total": self.opt.audio_length,
            # "padding_mask": 0,
            # "load_time": 1
            "windows_list": list(windows_list) if windows_list is not None else []
        }
        return audio_stereo, frames, metadata, audio_mono, audio_diff, audio_clips

    def __getitem__(self, index):
        return self.load_batch(index)

    def __len__(self):
        return len(self.audios)

    def name(self):
        return 'FAIR-PLAY'
