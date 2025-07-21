#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
import os.path
import random
from glob import glob

import h5py
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from loguru import logger
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


def rand_excl_z(z, start=0, end=99):
    options = [num for num in range(start, end + 1) if num != z]
    return random.choice(options)

def normalize_tensor(samples, desired_rms = 0.1, eps = 1e-4):
    rms = torch.maximum(torch.tensor(eps), torch.sqrt(torch.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return rms / desired_rms, samples

def generate_spectrogram(audio):
    spectro = librosa.core.stft(audio, n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return samples

class FAIRPLAY_DATSET(Dataset):
    def __init__(self, opt, mode=''):
        super().__init__()
        self.initialize(opt, mode)

    def initialize(self, opt, mode=''):
        self.opt = opt
        self.mode = mode

        self.audios = []
        if opt.setup == '10splits':
            #load hdf5 file here
            h5f_path = os.path.join(opt.data_root, "splits", opt.hdf5FolderPath, mode+".h5")
            logger.info(f"Loading {mode} data from <<< {h5f_path} >>>")
            h5f = h5py.File(h5f_path, 'r')
            self.audios = h5f['audio'][:]
            self.audios = [item.decode('UTF-8').split("/")[-1][:-4] for item in self.audios]
        elif opt.setup == '5splits':
            #load from txt file
            txt_path = os.path.join(opt.data_root, "five_splits", opt.hdf5FolderPath, mode+".txt")
            logger.info(f"Loading {mode} data from <<< {txt_path} >>>")
            with open(txt_path, 'r') as f:
                self.audios = [line.strip()[:-4] for line in f]
        else:
            raise NotImplementedError(f"Unknown setup: {opt.setup}")
        
        if mode == 'train':
            self.audios = self.audios * opt.data_vol_scaler

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.ToTensor(), normalize]
        self.vision_transform = transforms.Compose(vision_transform_list)
        self.mask_transform = transforms.Compose([transforms.ToTensor()])

        self.sample_rate = 16000
        self.hop_length = 512
        self.random_mono = opt.random_mono
        self.data_root = opt.data_root
        # 
        self.sample_size = opt.sample_size # 16384
        self.test_hop_size = opt.test_hop_size # 0.1
        self.samples_per_window = opt.sample_size # 16384

        self.multi_frames = opt.multi_frames

        self.type = opt.method_type
        if self.type == 'vae':
            # VAE
            self.samples_per_window = round((opt.audio_length * self.sample_rate) / 2048) * 2048
        elif self.type == 'm2b':
            # M2B
            self.samples_per_window = int(opt.audio_length * self.sample_rate)
        else:
            raise NotImplementedError(f'Unknown model type: {self.type}')

        logger.info(
            f"Test hop size: {self.test_hop_size}", 
            f"Samples per window: {self.samples_per_window}",
            f"Samples per window: {self.samples_per_window}"
        )
        self.use_mask = opt.use_mask

    def process_audio_augment(self, audio):
        audio = audio.copy().astype(np.float32)
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
    
    def prepare_audio(self, file_id):
        audio_file = os.path.join(self.data_root, "binaural_audios", file_id + ".wav")
        audio, audio_rate = librosa.load(audio_file, sr=self.sample_rate, mono=False)
        if self.mode == 'train':
            # randomly get a start time for the audio segment from the 10s clip
            audio_start_time = random.uniform(0, 9.9 - self.opt.audio_length)
            audio_end_time = audio_start_time + self.opt.audio_length
            audio_start = int(audio_start_time * self.sample_rate)
            audio_end = audio_start + int(self.opt.audio_length * self.sample_rate)
            
            if self.type == 'vae':
                # VAE
                pad_ = self.compute_padding((audio_end-audio_start), self.sample_rate)
                audio = audio[:, audio_start:(audio_end+pad_)]
            elif self.type == 'm2b':
                # M2B
                audio = audio[:, audio_start:audio_end]
                audio = normalize(audio)
            else:
                raise NotImplementedError(f'Unknown model type: {self.type}')

            return audio, (audio_start_time, audio_end_time), torch.empty(0), None
        else:
            audio = torch.tensor(audio)
            # unfold the audio into clips
            audio_clips, windows_list = self.generate_audio_clips(audio)
            return audio, None, audio_clips, windows_list
    
    def prepare_diff_audio(self, audio):
        audio_channel1 = audio[0, None]
        audio_channel2 = audio[1, None]

        audio_diff = audio_channel1 - audio_channel2
        audio_mono = audio_channel1 + audio_channel2

        return audio, audio_mono, audio_diff

    def prepare_last_frame(self, frame_path, frame_index):
        id_prev = frame_index - 1
        frames_prev = self._load_image_tta(frame_path, id_prev, augment=True)
        return frames_prev
    
    def _load_image(self, frame_path, frame_index):
        frames = Image.open(os.path.join(frame_path, str(frame_index).zfill(6) + '.png')).convert('RGB')
        frames = self.vision_transform(frames)
        return frames

    def _five_crop_points(self, w, h, crop_w=448,  crop_h=224):
        points = [
            (0, 0),  # Top-Left
            (w - crop_w, 0),  # Top-Right
            (0, h - crop_h),  # Bottom-Left
            (w - crop_w, h - crop_h),  # Bottom-Right
            ((w - crop_w) // 2, (h - crop_h) // 2)  # Center
        ]
        """ diagnal points """
        return points

    def _process_visual(self, image, frame_index, augment=False, crop_w=448,  crop_h=224):
        image = image.resize((480,240))
        w,h = image.size
        w_offset = w - crop_w
        h_offset = h - crop_h
        if self.mode == 'train':
            left = random.randrange(0, w_offset + 1)
            upper = random.randrange(0, h_offset + 1)
        else:
            points = self._five_crop_points(w, h, crop_w, crop_h)
            point_idx = frame_index % len(points)
            point_idx = max(0, min(point_idx, len(points) - 1))
            left, upper = points[point_idx]
            # image = image.resize((448,224))

        image = image.crop((left, upper, left+448, upper+224))

        if augment:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.random()*0.6 + 0.7)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.random()*0.6 + 0.7)
        return image

    def _process_image(self, image, frame_index, **kwargs):
        image = self._process_visual(image, frame_index, **kwargs)
        image = self.vision_transform(image)
        return image

    def _process_mask(self, mask, frame_index, **kwargs):
        mask = self._process_visual(mask, frame_index, augment=False, **kwargs)
        mask = self.mask_transform(mask)
        return mask

    def _load_image_tta(self, frame_path, frame_index, augment=False, crop_w=448,  crop_h=224):
        image_path = os.path.join(frame_path, str(frame_index).zfill(6) + '.png')
        image = Image.open(image_path).convert('RGB')
        image = self._process_image(image, frame_index, augment=augment, crop_w=crop_w, crop_h=crop_h)
        
        mask = torch.ones_like(image)
        if self.use_mask:
            mask_path = image_path.replace('/frames/', '/masks/')
            mask = Image.open(mask_path).convert('L')
            mask = self._process_mask(mask, frame_index, crop_w=crop_w, crop_h=crop_h)
        return image, mask
    
    def prepare_visual(self, file_id, clip_info=None, audio_len=None, windows_list=None, audio_num_samples=None):
        frame_path = os.path.join(self.data_root, 'frames', file_id + '.mp4')
        if self.mode == 'train':
            audio_start_time, audio_end_time = clip_info
            # get the closest frame to the audio segment
            # frame_index = int(round((audio_start_time + audio_end_time) / 2.0 + 0.5))  #1 frame extracted per second
            frame_index = int(round(((audio_start_time + audio_end_time) / 2.0 + self.test_hop_size) * 10))  #10 frames extracted per second
            frame_index = np.clip(frame_index, a_min=1, a_max=99)
            frames, masks = self._load_image_tta(frame_path, frame_index, augment=True)
            if self.multi_frames:
                # frames_prev = self.prepare_last_frame(frame_path, frame_index)
                prev_idx = frame_index - 1
                frames_prev, masks_prev = self._load_image_tta(frame_path, prev_idx, augment=True)
                frames = torch.stack([frames_prev, frames])
                masks = torch.stack([masks_prev, masks])
        else:
            num_frames = len(glob(os.path.join(frame_path, '*.png')))
            # frame_idx_list = [max(0, int((i/self.samples_per_window)*(num_frames/audio_len))-1) for i in windows_list]
            # ((i + self.samples_per_window / 2.0) / audio_num_samples) -> current audio second in terms of total
            frame_idx_list = [max(0, int(round((((i + self.samples_per_window / 2.0) / audio_num_samples) * audio_len + self.test_hop_size) * 10))) for i in windows_list]
            frames = [] 
            masks = []
            for i in frame_idx_list:
                if i > num_frames:
                    assert i < num_frames, f"Frame index {i} is greater than the number of frames {num_frames}"
                frame, mask = self._load_image_tta(frame_path, i, augment=False)
                frames.append(frame)
                masks.append(mask)
            frames = torch.stack(frames)
            masks = torch.stack(masks)
        return frames, masks
    
    def load_batch(self, index):
        file_id = self.audios[index]
        audio, clip_info, audio_clips, windows_list = self.prepare_audio(file_id)
        audio_stereo, audio_mono, audio_diff = self.prepare_diff_audio(audio)

        audio_stereo = torch.FloatTensor(audio_stereo)
        audio_mono = torch.FloatTensor(audio_mono)
        audio_diff = torch.FloatTensor(audio_diff)

        kwgs = {
            "clip_info": clip_info,
            "audio_len": int(audio.shape[-1]/self.sample_rate),
            "windows_list": windows_list,
            "audio_num_samples": audio.shape[-1],
        }
        frames, masks = self.prepare_visual(file_id, **kwgs)
        # return frames, audio_diff, audio_mono
        metadata = {
            "seconds_start": int(0.0), 
            "seconds_total": self.opt.audio_length,
            # "padding_mask": 0,
            # "load_time": 1
            "windows_list": list(windows_list) if windows_list is not None else [],
            "fn": file_id
        }
        return audio_stereo, frames, masks, metadata, audio_mono, audio_diff, audio_clips

    def __getitem__(self, index):
        return self.load_batch(index)

    def __len__(self):
        return len(self.audios)

    def name(self):
        return 'FAIR-PLAY'