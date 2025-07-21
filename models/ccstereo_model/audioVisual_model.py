import torch

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        self.net_visual, self.net_audio = nets
        self.flow_net = None

    def forward(self, input, volatile=False):
        flow_feature, prev_feat = None, None

        flow_input = input['flow']
        visual_input = input['frame']
        prev_frame = input['prev_frame']
        mask = input['mask']
        prev_mask = input['prev_mask']
        gt_diff_spec = input['audio_diff_spec']
        input_spectrogram = input['audio_mix_spec']
        visual_feature = self.net_visual(visual_input)

        if prev_frame is not None:
            prev_feat = self.net_visual(prev_frame)

        if flow_input is not None:
            flow_feature = self.flow_net(flow_input)
            
        mask_prediction, ssl_loss = self.net_audio(
            input_spectrogram, 
            visual_feature, 
            mask=mask,
            prev_mask=prev_mask, 
            prev_feat=prev_feat, 
            flow_feat=flow_feature
        )

        #complex masking to obtain the predicted spectrogram
        spectrogram_diff_real = input_spectrogram[:,0,:,:] * mask_prediction[:,0,:,:] - input_spectrogram[:,1,:,:] * mask_prediction[:,1,:,:]
        spectrogram_diff_img = input_spectrogram[:,0,:,:] * mask_prediction[:,1,:,:] + input_spectrogram[:,1,:,:] * mask_prediction[:,0,:,:]

        pred_diff_spec = torch.cat((spectrogram_diff_real.unsqueeze(1), spectrogram_diff_img.unsqueeze(1)), 1)

        output =  {
            'mask_prediction': mask_prediction, 
            'pred_diff_spec': pred_diff_spec, 
            'gt_diff_spec': gt_diff_spec,
            'ssl_loss': ssl_loss
        }
        return output
