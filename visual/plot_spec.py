import argparse
import os
from glob import glob

import librosa
import matplotlib.pyplot as plt
import torch
import torchaudio
from loguru import logger
from tqdm import tqdm

# define args

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = torch.maximum(torch.tensor(eps), torch.sqrt(torch.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return samples

def process_stft(audio, n_fft=512, hop_l=160, win_l=400):
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
    return audio_spec

class SpectrogramPlotter:
    def __init__(self, save_fp, num_rows=6):
        self.save_fp = save_fp
        self.num_rows = num_rows
        os.makedirs(save_fp, exist_ok=True)
        logger.info(f"Saving spectrograms to << {save_fp} >>")

    def get_mse_map(self, gt, pred):
        mse = (gt - pred) ** 2
        mse = (mse - mse.min()) / (mse.max() - mse.min())
        return mse

    def add_subplot(self, ax, subplot_, title="", fontsize=16, cmap='inferno'):
        ax.imshow(subplot_, origin='lower', aspect='auto', cmap=cmap)
        ax.set_title(title, fontsize=fontsize)
        ax.title.set_color('red')
        ax.set_xlabel("")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        return

    def plot_spectrograms(self, gt_spec, pred_spec, fn_, title=""):
        fig, axs = plt.subplots(self.num_rows, 1, figsize=(20, 20))
        gt_real = gt_spec[0]
        pred_real = pred_spec[0]
        mse_real = self.get_mse_map(gt_real, pred_real)

        gt_img = gt_spec[1]
        pred_img = pred_spec[1]
        mse_img = self.get_mse_map(gt_img, pred_img)

        titles = [
            f"{title}\nGT - Real",
            "Pred - Real",
            "MSE (MIN-MAX) - Real",
            "GT - IMG",
            "Pred - IMG",
            "MSE (MIN-MAX) - IMG"
        ]
        data = [gt_real, pred_real, mse_real, gt_img, pred_img, mse_img]
        cmaps = ['inferno', 'inferno', 'jet', 'inferno', 'inferno', 'jet']

        for i, ax in enumerate(axs):
            self.add_subplot(ax, data[i], title=titles[i], cmap=cmaps[i])
            
        plt.savefig(
            os.path.join(self.save_fp, f"{fn_}_spec.png"), 
            dpi=300, 
            bbox_inches='tight', 
            pad_inches=0.02
        )
        plt.cla()
    
    def _plot_channel(self, axs, gt_spec, pred_spec, title=""):
        gt_real = gt_spec[0]
        pred_real = pred_spec[0]
        mse_real = self.get_mse_map(gt_real, pred_real)

        gt_img = gt_spec[1]
        pred_img = pred_spec[1]
        mse_img = self.get_mse_map(gt_img, pred_img)

        titles = [
            f"{title}\nGT - Real",
            "Pred - Real",
            "MSE (MIN-MAX) - Real",
            "GT - IMG",
            "Pred - IMG",
            "MSE (MIN-MAX) - IMG"
        ]
        data = [gt_real, pred_real, mse_real, gt_img, pred_img, mse_img]
        cmaps = ['inferno', 'inferno', 'jet', 'inferno', 'inferno', 'jet']

        for i, ax in enumerate(axs):
            self.add_subplot(ax, data[i], title=titles[i], cmap=cmaps[i])

    def plot_stereo_spec(self, gt_spec_l, gt_spec_r, pred_spec_l, pred_spec_r, fn_, title=""):
        fig, axs = plt.subplots(self.num_rows, 2, figsize=(30, 20))

        self._plot_channel(axs[:,0], gt_spec_l, pred_spec_l, title="L")
        self._plot_channel(axs[:,1], gt_spec_r, pred_spec_r, title="R")
        plt.subplots_adjust(wspace=0.1, hspace=0.5) 
        plt.savefig(
            os.path.join(self.save_fp, f"{fn_}_spec.png"), 
            dpi=300, 
            bbox_inches='tight', 
            pad_inches=0.02
        )
        plt.cla()
    
    def plot_diff_spec(self, gt_spec_diff, pred_spec_diff, fn_, title=""):
        gt_spec = gt_spec_diff
        pred_spec = pred_spec_diff
        fig, axs = plt.subplots(self.num_rows, 1, figsize=(20, 20))

        gt_real = gt_spec[0]
        pred_real = pred_spec[0]
        mse_real = self.get_mse_map(gt_real, pred_real)

        gt_img = gt_spec[1]
        pred_img = pred_spec[1]
        mse_img = self.get_mse_map(gt_img, pred_img)

        axs[0].imshow(gt_real, origin='lower', aspect='auto', cmap='inferno')
        axs[1].imshow(pred_real, origin='lower', aspect='auto', cmap='inferno')
        axs[2].imshow(mse_real, origin='lower', aspect='auto', cmap='jet')

        axs[3].imshow(gt_img, origin='lower', aspect='auto', cmap='inferno')
        axs[4].imshow(pred_img, origin='lower', aspect='auto', cmap='inferno')
        axs[5].imshow(mse_img, origin='lower', aspect='auto', cmap='jet')

        axs[0].set_title(f"GT - Real ")
        axs[1].set_title("Pred - Real")
        axs[2].set_title("MSE (MIN-MAX) - Real")

        axs[3].set_title("GT - IMG")
        axs[4].set_title("Pred - IMG")
        axs[5].set_title("MSE (MIN-MAX) - IMG")
        # Change title to red
        for i in range(self.num_rows):
            axs[i].title.set_color('red')
            axs[i].set_xlabel("")
            axs[i].set_xticklabels([])
            axs[i].set_yticklabels([])
        plt.savefig(
            os.path.join(self.save_fp, f"{fn_}_spec.png"), 
            dpi=300, 
            bbox_inches='tight', 
            pad_inches=0.02
        )
        plt.cla()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="run-name")
    parser.add_argument("--save_root", type=str, default="./Test/spectrogram")
    parser.add_argument("--target_root", type=str, default="./Test/waveform")
    args = parser.parse_args()

    run_name = args.run_name
    save_fp = os.path.join(args.save_root, run_name)
    wave_fp = os.path.join(args.target_root, run_name)
    gt_fp = sorted(glob(os.path.join(wave_fp, "gt_diff", "*.wav")))
    pred_fp = sorted(glob(os.path.join(wave_fp, "pred_diff", "*.wav")))

    tbar = tqdm(zip(gt_fp, pred_fp), total=len(gt_fp))

    plotter = SpectrogramPlotter(save_fp)

    for gt, pred in tbar:
        assert gt.split("_")[-1] == pred.split("_")[-1]
        fn_ = gt.split("_")[-1][:-4]

        gt_waveform, _ = torchaudio.load(gt)
        pred_waveform, _ = torchaudio.load(pred)
        
        gt_spec = process_stft(gt_waveform)[0]
        pred_spec = process_stft(pred_waveform)[0]

        gt_spec = librosa.amplitude_to_db(gt_spec)
        pred_spec = librosa.amplitude_to_db(pred_spec)

        plotter.plot_spectrograms(gt_spec, pred_spec, fn_)
