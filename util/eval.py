import librosa
import numpy as np
from scipy.signal import hilbert
import torch
from util.audio_utils import FourierTransform
from einops import rearrange
from math import pi

def STFT_L2_distance(predicted_spect_channel1, gt_spect_channel1, predicted_spect_channel2, gt_spect_channel2):    #channel1
    real = np.expand_dims(np.real(predicted_spect_channel1), axis=0)
    imag = np.expand_dims(np.imag(predicted_spect_channel1), axis=0)
    predicted_realimag_channel1 = np.concatenate((real, imag), axis=0)
    real = np.expand_dims(np.real(gt_spect_channel1), axis=0)
    imag = np.expand_dims(np.imag(gt_spect_channel1), axis=0)
    gt_realimag_channel1 = np.concatenate((real, imag), axis=0)
    channel1_distance = np.mean(np.power((predicted_realimag_channel1 - gt_realimag_channel1), 2))

    #channel2
    real = np.expand_dims(np.real(predicted_spect_channel2), axis=0)
    imag = np.expand_dims(np.imag(predicted_spect_channel2), axis=0)
    predicted_realimag_channel2 = np.concatenate((real, imag), axis=0)
    real = np.expand_dims(np.real(gt_spect_channel2), axis=0)
    imag = np.expand_dims(np.imag(gt_spect_channel2), axis=0)
    gt_realimag_channel2 = np.concatenate((real, imag), axis=0)
    channel2_distance = np.mean(np.power((predicted_realimag_channel2 - gt_realimag_channel2), 2))

    #sum the distance between two channels
    stft_l2_distance = channel1_distance + channel2_distance
    return float(stft_l2_distance)

def Envelope_distance(predicted_binaural, gt_binaural):
    #channel1
    pred_env_channel1 = np.abs(hilbert(predicted_binaural[0,:]))
    gt_env_channel1 = np.abs(hilbert(gt_binaural[0,:]))
    channel1_distance = np.sqrt(np.mean((gt_env_channel1 - pred_env_channel1)**2))

    #channel2
    pred_env_channel2 = np.abs(hilbert(predicted_binaural[1,:]))
    gt_env_channel2 = np.abs(hilbert(gt_binaural[1,:]))
    channel2_distance = np.sqrt(np.mean((gt_env_channel2 - pred_env_channel2)**2))

    #sum the distance between two channels
    envelope_distance = channel1_distance + channel2_distance
    return float(envelope_distance)

def WAV_distance(predicted_binaural, gt_binaural):
    #channel1
    pred_env_channel1 = predicted_binaural[0,:]
    gt_env_channel1 = gt_binaural[0,:]
    channel1_distance = np.mean((gt_env_channel1 - pred_env_channel1)**2)

    #channel2
    pred_env_channel2 = predicted_binaural[1,:]
    gt_env_channel2 = gt_binaural[1,:]
    channel2_distance = np.mean((gt_env_channel2 - pred_env_channel2)**2)

    #sum the distance between two channels
    envelope_distance = channel1_distance + channel2_distance
    return float(envelope_distance)

def Envelope_distance(predicted_binaural, gt_binaural):
    #channel1
    pred_env_channel1 = np.abs(hilbert(predicted_binaural[0,:]))
    gt_env_channel1 = np.abs(hilbert(gt_binaural[0,:]))
    channel1_distance = np.sqrt(np.mean((gt_env_channel1 - pred_env_channel1)**2))

    #channel2
    pred_env_channel2 = np.abs(hilbert(predicted_binaural[1,:]))
    gt_env_channel2 = np.abs(hilbert(gt_binaural[1,:]))
    channel2_distance = np.sqrt(np.mean((gt_env_channel2 - pred_env_channel2)**2))

    #sum the distance between two channels
    envelope_distance = channel1_distance + channel2_distance
    return float(envelope_distance)

def SNR(predicted_binaural, gt_binaural):
    mse_distance = np.mean(np.power((predicted_binaural - gt_binaural), 2))
    snr = 10. * np.log10((np.mean(gt_binaural**2) + 1e-4) / (mse_distance + 1e-4))

    return float(snr)

def Magnitude_distance(predicted_spect_channel1, gt_spect_channel1, predicted_spect_channel2, gt_spect_channel2):
    stft_mse1 = np.mean(np.power(np.abs(predicted_spect_channel1 - gt_spect_channel1), 2))
    stft_mse2 = np.mean(np.power(np.abs(predicted_spect_channel2 - gt_spect_channel2), 2))

    return float(stft_mse1 + stft_mse2)

def Angle_Diff_distance(predicted_binaural, gt_binaural):
    gt_diff = gt_binaural[0] - gt_binaural[1]
    pred_diff = predicted_binaural[0] - predicted_binaural[1]
    gt_diff_spec = librosa.core.stft(gt_diff, n_fft=512, hop_length=160, win_length=400, center=True)
    pred_diff_spec = librosa.core.stft(pred_diff, n_fft=512, hop_length=160, win_length=400, center=True)
    _, pred_diff_phase = librosa.magphase(pred_diff_spec)
    _, gt_diff_phase = librosa.magphase(gt_diff_spec)
    pred_diff_angle = np.angle(pred_diff_phase)
    gt_diff_angle = np.angle(gt_diff_phase)
    angle_diff_init_distance = np.abs(pred_diff_angle - gt_diff_angle)
    angle_diff_distance = np.mean(np.minimum(angle_diff_init_distance, np.clip(2 * pi - angle_diff_init_distance, a_min=0, a_max=2*pi))) 
    
    return float(angle_diff_distance)

class Loss(torch.nn.Module):
    def __init__(self, mask_beginning=0):
        '''
        base class for losses that operate on the wave signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__()
        self.mask_beginning = mask_beginning

    def forward(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data = data[..., self.mask_beginning:]
        target = target[..., self.mask_beginning:]
        return self._loss(data, target)

    def _loss(self, data, target):
        pass


class L2Loss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        return torch.mean((data - target).pow(2))


class AmplitudeLoss(Loss):
    def __init__(self, sample_rate, mask_beginning=0):
        '''
        :param sample_rate: (int) sample rate of the audio signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__(mask_beginning)
        self.fft = FourierTransform(
            fft_bins = 512,
            win_length = 400,
            hop_length = 160,
            sample_rate=sample_rate,
            return_complex=True
        )

    def _transform(self, data):
        return self.fft.stft(data.view(-1, data.shape[-1]))

    def _loss(self, data, target, eps=1e-12):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data, target = self._transform(data), self._transform(target)
        data = torch.maximum(torch.tensor(eps), torch.sum(data**2, dim=-1)) ** 0.5
        target = torch.maximum(torch.tensor(eps), torch.sum(target**2, dim=-1)) ** 0.5
        return torch.mean(torch.abs(data - target))


class PhaseLoss(Loss):
    def __init__(self, sample_rate, mask_beginning=0, ignore_below=0.1):
        '''
        :param sample_rate: (int) sample rate of the audio signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__(mask_beginning)
        self.ignore_below = ignore_below
        self.fft = FourierTransform(
            fft_bins = 512,
            win_length = 400,
            hop_length = 160,
            sample_rate=sample_rate,
            return_complex=True
        )

    def _transform(self, data):
        return self.fft.stft(data.view(-1, data.shape[-1]))

    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data, target = self._transform(data).view(-1, 2), self._transform(target).view(-1, 2) # [B, 257, 1001, 2]
        # ignore low energy components for numerical stability
        target_energy = torch.sum(torch.abs(target), dim=-1)
        pred_energy = torch.sum(torch.abs(data.detach()), dim=-1)
        target_mask = target_energy > self.ignore_below * torch.mean(target_energy)
        pred_mask = pred_energy > self.ignore_below * torch.mean(target_energy)
        indices = torch.nonzero(target_mask * pred_mask).view(-1)
        data, target = torch.index_select(data, 0, indices), torch.index_select(target, 0, indices)
        # compute actual phase loss in angular space
        data_angles, target_angles = torch.atan2(data[:, 0], data[:, 1]), torch.atan2(target[:, 0], target[:, 1])
        loss = torch.abs(data_angles - target_angles)
        # positive + negative values in left part of coordinate system cause angles > pi
        # => 2pi -> 0, 3/4pi -> 1/2pi, ... (triangle function over [0, 2pi] with peak at pi)
        loss = np.pi - torch.abs(loss - np.pi)
        return torch.mean(loss)
    
def spec_amp_loss(data, target):
    '''
    :param data: predicted wave signals in a B x channels x T tensor
    :param target: target wave signals in a B x channels x T tensor
    :return: a scalar loss value
    '''
    data = torch.sum(data**2, dim=-1) ** 0.5
    target = torch.sum(target**2, dim=-1) ** 0.5
    return torch.mean(torch.abs(data - target))

def spec_phase_loss(data, target, ignore_below=0.2):
    b, c, f, t = data.shape
    data = rearrange(data, 'b c f t -> (b f t) c')
    target = rearrange(target, 'b c f t -> (b f t) c')
    # ignore low energy components for numerical stability
    target_energy = torch.sum(torch.abs(target), dim=-1)
    pred_energy = torch.sum(torch.abs(data.detach()), dim=-1)
    target_mask = target_energy > ignore_below * torch.mean(target_energy)
    pred_mask = pred_energy > ignore_below * torch.mean(target_energy)
    indices = torch.nonzero(target_mask * pred_mask).contiguous().view(-1)
    data, target = torch.index_select(data, 0, indices), torch.index_select(target, 0, indices)
    # compute actual phase loss in angular space
    data_angles, target_angles = torch.atan2(data[:, 0], data[:, 1]), torch.atan2(target[:, 0], target[:, 1])
    loss = torch.abs(data_angles - target_angles)
    # positive + negative values in left part of coordinate system cause angles > pi
    # => 2pi -> 0, 3/4pi -> 1/2pi, ... (triangle function over [0, 2pi] with peak at pi)
    loss = np.pi - torch.abs(loss - np.pi)
    return torch.mean(loss)
