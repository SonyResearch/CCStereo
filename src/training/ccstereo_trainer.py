
import os
import typing as tp

import librosa
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio
import wandb
from einops import rearrange
from loguru import logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from safetensors.torch import save_model
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from util.eval import (
    SNR,
    AmplitudeLoss,
    Angle_Diff_distance,
    Envelope_distance,
    L2Loss,
    Magnitude_distance,
    PhaseLoss,
    STFT_L2_distance,
    spec_amp_loss,
)

from .logging import MetricsLogger
from .losses.auraloss import MultiResolutionSTFTLoss
from .scheduler import create_scheduler_from_config

def normalize(samples):
    return samples / torch.maximum(torch.tensor(1e-20), torch.max(torch.abs(samples)))

def audio_normalize(samples, desired_rms = 0.1, eps = 1e-4):
    # rms = torch.maximum(torch.tensor(eps), torch.sqrt(torch.mean(samples**2)))
    rms = torch.maximum(
        torch.tensor(1e-4),
        torch.sqrt(torch.mean(samples**2, dim=-1, keepdim=True))
    )
    samples = samples * (desired_rms / rms)
    return rms / desired_rms, samples

def process_istft(spec, aud_len, n_fft=512, hop_l=160, win_l=400, eps=1e-12):
    spec = spec - eps
    window_fn = torch.hann_window(win_l, device=spec.device)
    kwgs = {
        'n_fft': n_fft, 
        'hop_length': hop_l, 
        'win_length': win_l,
        'center': True,
        'length': aud_len,
        'return_complex': False,
        'window': window_fn
    }
    result = torch.istft(spec, **kwgs)
    return result

def process_stft(audio, n_fft=512, hop_l=160, win_l=400, eps=1e-12):
    audio = audio.squeeze(1)
    window_fn = torch.hann_window(win_l, device=audio.device)
    kwgs = {
        'n_fft': n_fft, 
        'hop_length': hop_l, 
        'win_length': win_l,
        'center': True,
        'pad_mode': 'constant',
        'return_complex': True,
        'window': window_fn
    }
    audio_spec = torch.stft(audio, **kwgs)
    audio_spec = torch.stack((audio_spec.real, audio_spec.imag), dim=1)
    return audio_spec + eps

def diff_to_stereo(audio_mono: torch.Tensor, decoded: torch.Tensor, normalizer: torch.Tensor = None):
    decoded = decoded[:,0,:,:] + (1j * decoded[:,1,:,:])
    decoded = process_istft(decoded, aud_len=audio_mono.shape[-1])
    decoded = decoded.unsqueeze(1)

    fakes_l = (audio_mono + decoded) / 2
    fakes_r = (audio_mono - decoded) / 2
    fakes = torch.cat([fakes_l, fakes_r], dim=1)

    if normalizer is not None:
        fakes = fakes * normalizer
    return fakes

class AutoencoderTrainingWrapper(pl.LightningModule):
    def __init__(
        self,
        autoencoder: nn.Module,
        loss_config: dict,
        optimizer_configs: dict,
        lr: float = 1e-4,
        warmup_steps: int = 0,
        encoder_freeze_on_warmup: bool = False,
        sample_rate: int = 48000,
        use_ema: bool = True,
        ema_copy=None,
        force_input_mono: bool = False,
        latent_mask_ratio: float = 0.0,
        teacher_model: tp.Optional[nn.Module] = None,
        logging_config: dict = {},
        test_hop_size: float = 0.1,
        #
        train_config: dict = {},
    ):
        super().__init__()

        self.sample_rate = sample_rate

        self.automatic_optimization = False
        self.autoencoder = autoencoder
        self.teacher_model = teacher_model
        
        self.warmup_steps = warmup_steps
        self.encoder_freeze_on_warmup = encoder_freeze_on_warmup
        self.force_input_mono = force_input_mono
        self.latent_mask_ratio = latent_mask_ratio

        self.optimizer_configs = optimizer_configs
        self.loss_config = loss_config

        self.log_every = logging_config.get("log_every", 1)

        self.metrics_logger = MetricsLogger()
        self.mse_loss = torch.nn.MSELoss()

        self.best_stft_l2_dist = 1000
        self.wav_loss = L2Loss()
        self.amp_loss = AmplitudeLoss(sample_rate=self.sample_rate)
        self.phase_loss = PhaseLoss(sample_rate=self.sample_rate, ignore_below=0.2)
        
        self.test_hop_size = test_hop_size
        self.use_ema = use_ema

        self.save_visual_root = None
        self.checkpoint_dir = None
        self.spec_plotter = None
        self.eval_list = []

        self.epochs = train_config["epochs"]
        self.epoch_inters = train_config["epoch_inters"]
        self.use_flow = train_config.get("use_flow", False)
        self.multi_frames = train_config.get("multi_frames", False)
        self.use_mask = train_config.get("use_mask", False)

        self.lr_img = train_config.get("lr_img", 5e-5)
        self.lr_aud = train_config.get("lr_audio", 5e-4)
        self.weight_decay = train_config.get("weight_decay", 5e-4)

        stft_loss_args = loss_config['spectral']['config']
        self.sdstft = MultiResolutionSTFTLoss(sample_rate=sample_rate, **stft_loss_args)
        
        self.flow_model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
        for param in self.flow_model.parameters():
            param.requires_grad = False
        self.flow_model = self.flow_model.eval()

        self.mel_scale = torchaudio.transforms.MelScale(
                n_mels=64, sample_rate=self.sample_rate, n_stft=257)
        self.eps = 1e-8

    def configure_optimizers(self):
        param_groups = [{'params': self.autoencoder.net_visual.parameters(), 'lr': self.lr_img},
                        {'params': self.autoencoder.net_audio.parameters(), 'lr': self.lr_aud}]
        opt_gen = torch.optim.Adam(param_groups, betas=(0.9, 0.999), weight_decay=self.weight_decay)

        scheduler = {
            "type": "PolynomialLR",
            "config": {
                "total_iters": self.epochs * self.epoch_inters,
                "power": 0.9,
            }
        }
        sched_gen = create_scheduler_from_config(scheduler, opt_gen)
        return [opt_gen], [sched_gen]

    def stft_to_mel_scale(self, spec):
        power_spec = spec.norm(p=2, dim=1)
        mel_spec = self.mel_scale(power_spec)
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=self.eps))
        return log_mel_spec.unsqueeze(1)

    def training_step(self, batch, batch_idx):

        reals, frames, masks, _, audio_mono, audio_diff, _ = batch

        reals = reals.to(self.device)
        frames = frames.to(self.device)
        masks = masks.to(self.device)
        audio_mono = audio_mono.to(self.device)
        audio_diff = audio_diff.to(self.device)

        flow_info = None
        if self.multi_frames:
            prev_frame, input_frame = frames[:,0], frames[:,1]
            prev_mask, input_mask = masks[:,0], masks[:,1]
            if self.use_flow:
                flow_info = self.get_flow_feature(prev_frame, input_mask, split=8)
        else:
            input_frame = frames
            input_mask = masks

        audio_diff_spec = process_stft(audio_diff)
        audio_mix_spec = process_stft(audio_mono)

        input_data = {
            'frame': input_frame,
            'prev_frame': prev_frame if self.multi_frames else None,
            'mask': input_mask if self.use_mask else None,
            'prev_mask': prev_mask if self.multi_frames & self.use_mask else None,
            'flow': flow_info,
            'audio_diff_spec': audio_diff_spec,
            'audio_mix_spec': audio_mix_spec,
        }

        output = self.autoencoder.forward(input_data)
        pred = output['pred_diff_spec']
        gt = output['gt_diff_spec']
        
        """ 
        LOSS FUNCTIONS 
        """
        ms_ssim_loss = torch.tensor(0.0, device=self.device)
        amp_loss = torch.tensor(0.0, device=self.device)
        phase_loss = torch.tensor(0.0, device=self.device)
        cpaps_loss = torch.tensor(0.0, device=self.device)
        phase_wav_loss = torch.tensor(0.0, device=self.device)
        amp_wav_loss = torch.tensor(0.0, device=self.device)
        ssl_loss = torch.tensor(0.0, device=self.device)
        ssl_loss = output['ssl_loss'] * 0.1
        mse_loss = self.mse_loss(pred, gt)
        amp_loss = spec_amp_loss(pred, gt) * 0.005
        fakes = diff_to_stereo(audio_mono, pred)
        phase_wav_loss = self.phase_loss(reals, fakes)

        loss = \
            mse_loss + \
            amp_loss + amp_wav_loss + \
            phase_loss + phase_wav_loss + \
            ssl_loss 

        opt_gen = self.optimizers()
        sched_gen = self.lr_schedulers()

        opt_gen.zero_grad()
        self.manual_backward(loss)
        opt_gen.step()
        sched_gen.step()

        log_dict = {
            'train/mse_loss': mse_loss.detach(),
            'train/amp_loss': amp_loss.detach(),
            'train/amp_wav_loss': amp_wav_loss.detach(),
            'train/phase_loss': phase_loss.detach(),
            'train/phase_wav_loss': phase_wav_loss.detach(),
            'train/ms_ssim_loss': ms_ssim_loss.detach(),
            'train/cpaps_loss': cpaps_loss.detach(),
            'train/ssl_loss': ssl_loss.detach(),
            'train/visual_lr': opt_gen.param_groups[0]['lr'],
            'train/audio_lr': opt_gen.param_groups[1]['lr']
        }

        self.metrics_logger.add(log_dict)
        log_dict = self.metrics_logger.pop()
        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss

    def integrate_with_sliding_window(self, fakes, demo_size, windows_list, samples_per_window):
        n_chunk = len(windows_list)
        bs, ch, demo_len = demo_size
        hopsize_s = int(samples_per_window * self.test_hop_size)
        len_ = int(hopsize_s * n_chunk) + samples_per_window

        overlap_count = torch.zeros((bs, ch, len_), device=self.device) 
        binaural_audio = torch.zeros((bs, ch, len_), device=self.device)

        for idx, sliding_window_start in enumerate(windows_list):
            sliding_window_start = int(sliding_window_start)
            sliding_window_end = sliding_window_start + samples_per_window
            binaural_audio[:,:,sliding_window_start:sliding_window_end] += fakes[[idx], ...]
            overlap_count[:,:,sliding_window_start:sliding_window_end] += 1

        predicted_binaural = torch.divide(binaural_audio, overlap_count).squeeze(0).cpu()
        predicted_binaural = predicted_binaural[:, :demo_size[-1]]
        return predicted_binaural

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        return self.eval_model(batch, batch_idx, save_audio=False, mode='test')  

    def on_validation_epoch_end(self):
        mode = 'test'
        stft_l2_dist_ = self.trainer.callback_metrics[f"{mode}/stft_l2_dist_epoch"]
        env_dist_ = self.trainer.callback_metrics[f"{mode}/env_dist_epoch"]
        wav_dist_ = self.trainer.callback_metrics[f"{mode}/wav_dist_epoch"]
        amp_dist_ = self.trainer.callback_metrics[f"{mode}/amp_dist_epoch"]
        phase_dist_ = self.trainer.callback_metrics[f"{mode}/phase_dist_epoch"]
        snr_ = self.trainer.callback_metrics[f"{mode}/snr_epoch"]
        mag_dist_ = self.trainer.callback_metrics[f"{mode}/mag_dist_epoch"]
        ang_diff = self.trainer.callback_metrics[f"{mode}/ang_diff_epoch"]

        logger.success(f"Testing STFT L2 Dist: {stft_l2_dist_}")

        if stft_l2_dist_ < self.best_stft_l2_dist:
            self.best_stft_l2_dist = stft_l2_dist_
            wandb.run.summary["best_epoch"] = self.current_epoch
            wandb.run.summary[f"best_{mode}_stft_l2_dist"] = stft_l2_dist_
            wandb.run.summary[f"best_{mode}_env_dist"] = env_dist_
            wandb.run.summary[f"best_{mode}_wav_dist"] = wav_dist_
            wandb.run.summary[f"best_{mode}_amp_dist"] = amp_dist_
            wandb.run.summary[f"best_{mode}_phase_dist"] = phase_dist_
            wandb.run.summary[f"best_{mode}_snr"] = snr_
            wandb.run.summary[f"best_{mode}_mag_dist"] = mag_dist_
            wandb.run.summary[f"best_{mode}_ang_diff"] = ang_diff

            model = self.autoencoder
            path_ = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save({"state_dict": model.state_dict()}, path_)

        return 

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return self.eval_model(batch, batch_idx, save_audio=True, mode='test')

    def get_results(self, pred, gt, mode='test'):
        pred_np = pred.cpu().numpy()
        gt_np = gt.cpu().numpy()
        #
        pred_spec_ch1 = librosa.core.stft(
            pred_np[0,:], n_fft=512, hop_length=160, win_length=400, center=True)
        pred_spec_ch2 = librosa.core.stft(
            pred_np[1,:], n_fft=512, hop_length=160, win_length=400, center=True)
        #
        gt_spec_ch1 = librosa.core.stft(
            gt_np[0,:], n_fft=512, hop_length=160, win_length=400, center=True)
        gt_spec_ch2 = librosa.core.stft(
            gt_np[1,:], n_fft=512, hop_length=160, win_length=400, center=True)
        #
        stft_l2_dist_ = STFT_L2_distance(pred_spec_ch1, gt_spec_ch1, pred_spec_ch2, gt_spec_ch2)
        env_dist_ = Envelope_distance(pred_np, gt_np)
        snr_ = SNR(pred_np, gt_np)
        mag_dist_ = Magnitude_distance(pred_spec_ch1, gt_spec_ch1, pred_spec_ch2, gt_spec_ch2)
        ang_diff = Angle_Diff_distance(pred_np, gt_np)

        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
        # wav_dist_ = self.wav_loss(pred, gt) * 1e3    
        wav_dist_ = self.wav_loss(pred[:,0,None], gt[:,0,None]) + \
                    self.wav_loss(pred[:,1,None], gt[:,1,None])
        wav_dist_ = wav_dist_ * 1e3
        # amp_dist_ = self.amp_loss(pred, gt)
        amp_dist_ = self.amp_loss(pred[:,0,None], gt[:,0,None]) + \
                    self.amp_loss(pred[:,1,None], gt[:,1,None])
        # phase_dist_ = self.phase_loss(pred, gt)
        phase_dist_ = self.phase_loss(pred[:,0,None], gt[:,0,None]) + \
                      self.phase_loss(pred[:,1,None], gt[:,1,None])
        
        results_ = {
            f"{mode}/stft_l2_dist": stft_l2_dist_, 
            f"{mode}/env_dist": env_dist_,
            f"{mode}/wav_dist": wav_dist_,
            f"{mode}/amp_dist": amp_dist_,
            f"{mode}/phase_dist": phase_dist_,
            #
            f"{mode}/snr": snr_,
            f"{mode}/mag_dist": mag_dist_,
            f"{mode}/ang_diff": ang_diff
        }

        return results_
    
    def get_flow_feature(self, prev_frame, frames, split=2):
        B, C, H, W = frames.shape
        prev_frame = rearrange(prev_frame, '(n b) c h w -> n b c h w', n=split)
        frames = rearrange(frames, '(n b) c h w -> n b c h w', n=split)
        fea_list = []
        for i in range(split):
            flow_info = self.flow_model(prev_frame[i], frames[i])[-1]
            fea_list.append(flow_info)
        return torch.cat(fea_list, dim=0)

    @torch.no_grad()
    def eval_model(self, batch, _, save_audio=False, mode='test'):

        reals, frames, masks, demo_cond, audio_mono, audio_diff, audio_clips = batch
        file_id = demo_cond[0]['fn'].split('.')[0]

        reals = reals.to(self.device)
        frames = frames.to(self.device)
        masks = masks.to(self.device)
        audio_mono = audio_mono.to(self.device)
        audio_diff = audio_diff.to(self.device)
        audio_clips = audio_clips.to(self.device)

        audio_clips = audio_clips[0]
        frames = frames[0]
        masks = masks[0]
        
        gt_diff = audio_clips[:, 0, None] - audio_clips[:, 1, None]
        mono_clips = audio_clips[:, 0, None] + audio_clips[:, 1, None]

        normalizer, mono_clips = audio_normalize(mono_clips)
        
        audio_mix_spec = process_stft(mono_clips)
        audio_diff_spec = process_stft(gt_diff)

        flow_info = None

        input_data = {
            'frame': frames,
            'mask': masks if self.use_mask else None,
            'prev_mask': None,
            'prev_frame': None,
            'flow': flow_info,
            'audio_diff_spec': audio_diff_spec,
            'audio_mix_spec': audio_mix_spec,
        }
        output = self.autoencoder.forward(input_data)
        decoded = output['pred_diff_spec']

        fakes = diff_to_stereo(mono_clips, decoded, normalizer)

        demo_size = reals.shape
        #
        windows_list = demo_cond[0]["windows_list"]
        
        _, _, sample_size = audio_clips.shape
        predicted_binaural = self.integrate_with_sliding_window(fakes, demo_size, windows_list, sample_size)
        gt_binaural = reals.squeeze(0).cpu()
        
        predicted_binaural = normalize(predicted_binaural)
        gt_binaural = normalize(gt_binaural)

        results_ = self.get_results(predicted_binaural, gt_binaural, mode=mode)
        self.log_dict(results_, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        return results_
        
    def export_model(self, path, use_safetensors=False):
        model = self.autoencoder
        if use_safetensors:
            save_model(model, path)
        else:
            torch.save({"state_dict": model.state_dict()}, path)

    @property
    def warmed_up(self) -> bool:
        return self.global_step >= self.warmup_steps


class AutoencoderDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_dl,
        demo_every=2000,
        max_num_sample=4,
        sample_size=65536,
        sample_rate=48000,
        train_config={},
    ):
        super().__init__()
        self.demo_every = demo_every
        self.max_num_sample = max_num_sample
        self.demo_samples = sample_size
        self.demo_dl = iter(demo_dl)
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.use_flow = train_config.get("use_flow", False)
        self.multi_frames = train_config.get("multi_frames", False)
        self.use_mask = train_config.get("use_mask", False)
    
    def upload_audio(self, log_dict, audio, train_log_dir="", caption="recon"):
        filename = os.path.join(train_log_dir, f'{caption}_audio.wav')
        audio = audio.to(torch.float32).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        torchaudio.save(filename, audio, self.sample_rate)
        log_dict[caption] = wandb.Audio(filename, sample_rate=self.sample_rate, caption=caption)
        return log_dict, audio
