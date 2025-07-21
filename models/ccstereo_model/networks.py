#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.ccstereo_model.avad_layer import AVAD

from .transformer_decoder import CrossAttentionLayer, FFNLayer, SelfAttentionLayer


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])

class downconv_block(nn.Module):
    def __init__(self, input_nc, output_nc, audio_size=(16, 4), img_dim=512, norm_layer=nn.BatchNorm2d, use_avad=False):
        super(downconv_block, self).__init__()
        self.downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
        self.downrelu = nn.LeakyReLU(0.2, True)
        self.use_avad = use_avad
        if self.use_avad:
            self.downnorm = AVAD(output_nc, audio_size, vision_channel=img_dim)
        else:
            self.downnorm = norm_layer(output_nc)
        
    def forward(self, x, img_feat=None):
        x = self.downconv(x)
        x = self.downrelu(x)
        if self.use_avad:
            x = self.downnorm(x, img_feat)
        else:
            x = self.downnorm(x)
        return x

class upconv_block(nn.Module):
    def __init__(self, input_nc, output_nc, audio_size=(16, 4), img_dim=512, norm_layer=nn.BatchNorm2d, outermost=False, use_avad=False):
        super(upconv_block, self).__init__()
        self.upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
        self.uprelu = nn.ReLU(True)
        if not outermost:
            if use_avad:
                self.upnorm = AVAD(output_nc, audio_size, vision_channel=img_dim)
            else:
                self.upnorm = norm_layer(output_nc)
        self.outermost = outermost
        self.sigmoid = nn.Sigmoid()
        self.use_avad = use_avad

    def forward(self, x, img_feat):
        x = self.upconv(x)

        if self.outermost:
            x = self.sigmoid(x)
            return x
        if self.use_avad:
            x = self.upnorm(x, img_feat)
        else:
            x = self.upnorm(x)
        x = self.uprelu(x)
        return x
        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        if m.affine:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

def contrastive_triplet_loss_cosine(anchor_fea, pos_fea, neg_fea, temperature=0.1):
    """
    Compute the contrastive triplet loss using cosine similarity.
    
    Parameters:
    - anchor_fea: Tensor of shape [B, D] - Anchor features
    - pos_fea: Tensor of shape [B, D] - Positive features (similar to anchor)
    - neg_fea: Tensor of shape [B, D] - Negative features (different from anchor)
    - temperature: float - Temperature scaling parameter for similarity
    
    Returns:
    - loss: Tensor - The computed triplet loss using cosine similarity
    """
    anchor_fea = F.normalize(anchor_fea, dim=1)
    pos_fea = F.normalize(pos_fea, dim=1)
    neg_fea = F.normalize(neg_fea, dim=1)
    b = anchor_fea.shape[0]
    labels = torch.arange(b).to(anchor_fea.device).view(-1, 1)
    labels = torch.cat([labels, labels, labels+10000], dim=0)
    anchors_ = torch.cat([anchor_fea, pos_fea, neg_fea], dim=0)
    contras_ = anchors_.clone()
    loss = info_nce(anchors_, labels, contras_, labels, temperature=0.1, eps=1e-8)
    return loss

def info_nce(anchors_, a_labels_, contras_, c_labels_, temperature=0.1, eps=1e-8):
    # calculates the binary mask: same category => 1, different categories => 0
    mask = torch.eq(a_labels_, torch.transpose(c_labels_, 0, 1)).float()
    # calculates the dot product
    anchor_dot_contrast = torch.div(torch.matmul(anchors_, torch.transpose(contras_, 0, 1)),
                                    temperature)

    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # calculates the negative mask
    neg_mask = 1 - mask

    # avoid the self duplicate issue
    mask = mask.fill_diagonal_(0.)

    # sum the negative odot results
    neg_logits = torch.exp(logits) * neg_mask
    neg_logits = neg_logits.sum(1, keepdim=True)

    exp_logits = torch.exp(logits)
    
    # logits -> log(exp(x*x.T))
    # log_prob -> log(exp(x))-log(exp(x) + exp(y))
    # log_prob -> log{exp(x)/[exp(x)+exp(y)]}
    log_prob = logits - torch.log(exp_logits + neg_logits)
    assert ~torch.isnan(log_prob).any(), "nan check 1."

    # calculate the info-nce based on the positive samples (under same categories)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+eps)
    assert ~torch.isnan(mean_log_prob_pos).any(), "nan check 2."

    return - mean_log_prob_pos.mean()

class VisualNet(nn.Module):
    def __init__(self, original_resnet):
        super(VisualNet, self).__init__()
        layers = list(original_resnet.children())[0:-2]
        self.feature_extraction = nn.Sequential(*layers) #features before conv1x1

    def forward(self, x):
        # x = self.feature_extraction(x)
        fea = []
        for idx, layer in enumerate(self.feature_extraction):
            x = layer(x)
            if idx >=4:
                fea.append(x)
        return fea

class AudioNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2, img_dim=512, tg=[True, False, False, False]):
        super(AudioNet, self).__init__()

        v_h, v_w = 7, 14
        a_h, a_w = 8, 2
        hidden_dim = img_dim
        nheads = 8
        pre_norm = False
        dim_feedforward = img_dim
        self.img_pos_embed = nn.Embedding(v_h*v_w, hidden_dim)
        self.aud_pos_embed = nn.Embedding(a_h*a_w, hidden_dim)
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.transformer_self_attention_layers.append(
            SelfAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=0.0,
                normalize_before=pre_norm,
            )
        )
        self.transformer_cross_attention_layers.append(
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=0.0,
                normalize_before=pre_norm,
            )
        )
        self.transformer_ffn_layers.append(
            FFNLayer(
                d_model=hidden_dim,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
                normalize_before=pre_norm,
            )
        )
        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = upconv_block(ngf * 8, ngf * 8, audio_size=(16, 4), img_dim=img_dim, use_avad=tg[0]) #1296 (audio-visual feature) = 784 (visual feature) + 512 (audio feature)
        self.audionet_upconvlayer2 = upconv_block(ngf * 16, ngf * 4, audio_size=(32, 8), img_dim=img_dim, use_avad=tg[1])
        self.audionet_upconvlayer3 = upconv_block(ngf * 8, ngf * 2, audio_size=(64, 16), img_dim=img_dim, use_avad=tg[2])
        self.audionet_upconvlayer4 = upconv_block(ngf * 4, ngf, audio_size=(128, 32), img_dim=img_dim, use_avad=tg[3])
        self.audionet_upconvlayer5 = upconv_block(ngf * 2, output_nc, audio_size=(256, 64), img_dim=img_dim, outermost=True) #outermost layer use a sigmoid to bound the mask
        self.conv1x1_rgb = create_conv(img_dim, img_dim, 1, 0) #reduce dimension of extracted visual features
        self.cos_sim = nn.CosineSimilarity(dim=1)
    
    def interpolation(self, x, t):
        return F.interpolate(x, t.shape[-2:], mode='bilinear', align_corners=True)

    def forward_attn_fusion(self, visual_fea, audio_fea, mask=None):
        b, d, h, w = visual_fea.shape
        _, _, ah, aw = audio_fea.shape
        visual_fea = self.conv1x1_rgb(visual_fea)
        visual_fea = rearrange(visual_fea, 'b d h w -> (h w) b d')
        audio_fea = rearrange(audio_fea, 'b d h w -> (h w) b d')
        img_pos_embed = self.img_pos_embed.weight.unsqueeze(1).repeat(1, b, 1)
        query_embed = self.aud_pos_embed.weight.unsqueeze(1).repeat(1, b, 1)
        output = self.transformer_cross_attention_layers[0](
            audio_fea, visual_fea,
            memory_mask=None,
            memory_key_padding_mask=None,  # here we do not apply masking on padded region
            pos=img_pos_embed, query_pos=query_embed
        )
        output = self.transformer_self_attention_layers[0](
            output, tgt_mask=None,
            tgt_key_padding_mask=None,
            query_pos=query_embed
        )
        # FFN
        output = self.transformer_ffn_layers[0](output)

        output = rearrange(output, '(h w) b d -> b d h w', h=ah, w=aw)
        return output

    def forward_fusion(self, visual_feat, audio_fea):
        # audio_fea_shape -> [B, 512, 8, 3 ]
        # visual_feat     -> [B, 512, 7, 14]
        visual_feat = self.conv1x1_rgb(visual_feat)
        v_fea = visual_feat.clone()
        visual_feat = visual_feat.view(visual_feat.shape[0], -1, 1, 1) #flatten visual feature
        visual_feat = visual_feat.repeat(1, 1, audio_fea.shape[-2], audio_fea.shape[-1]) #tile visual feature
        audioVisual_feature = torch.cat((visual_feat, audio_fea), dim=1)
        return audioVisual_feature, v_fea
    
    def forward_fea_shuffle(self, v):
        b, d, h, w = v.shape
        v = rearrange(v, "b d h w -> b d (h w)")
        new_order = torch.randperm(v.shape[-1])
        v = v[:, :, new_order]
        v = rearrange(v, "b d (h w) -> b d h w", h=h, w=w)
        return v
    
    def cos_loss(self, z1, z2):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        return (self.cos_sim(z1, z2.detach()).mean() \
                + self.cos_sim(z2, z1.detach()).mean()) * 0.5

    def forward(self, x, visual_feat, mask=None, prev_feat=None, **kwargs):
        ssl_loss = torch.tensor(0.0, device=x.device)

        """ Unet encoder forward pass """
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature) 

        """ audio-visual fusion """
        audioVisual_feature = self.forward_attn_fusion(visual_feat, audio_conv5feature, mask=mask)

        if prev_feat is not None:
            flipped_visual = self.forward_fea_shuffle(visual_feat)
            anchor_fea = audioVisual_feature.clone()
            pos_fea = self.forward_attn_fusion(prev_feat, audio_conv5feature)
            neg_fea = self.forward_attn_fusion(flipped_visual, audio_conv5feature)
            anchor_fea = F.adaptive_avg_pool2d(anchor_fea, (1, 1)).squeeze() # [B, D]
            pos_fea = F.adaptive_avg_pool2d(pos_fea, (1, 1)).squeeze() # [B, D]
            neg_fea = F.adaptive_avg_pool2d(neg_fea, (1, 1)).squeeze() # [B, D]
            ssl_loss = contrastive_triplet_loss_cosine(anchor_fea, pos_fea, neg_fea)
        
        """ Unet decoder forward pass """
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature, visual_feat)
        audio_conv4feature = self.interpolation(audio_conv4feature, audio_upconv1feature)
        audio_upconv2feature = self.audionet_upconvlayer2(
            torch.cat((audio_upconv1feature, audio_conv4feature), dim=1), visual_feat
        )
        audio_conv3feature = self.interpolation(audio_conv3feature, audio_upconv2feature)
        audio_upconv3feature = self.audionet_upconvlayer3(
            torch.cat((audio_upconv2feature, audio_conv3feature), dim=1), visual_feat
        )
        audio_conv2feature = self.interpolation(audio_conv2feature, audio_upconv3feature)
        audio_upconv4feature = self.audionet_upconvlayer4(
            torch.cat((audio_upconv3feature, audio_conv2feature), dim=1), visual_feat
        )
        audio_conv1feature = self.interpolation(audio_conv1feature, audio_upconv4feature)

        """ 
        Assumes that the mask is for left channel 
        [A_M * M_L, A_M * (1 - M_L)]

        then, difference between left and right channel is calculated: 
        A_M * M_L - A_M * (1 - M_L) 
          = 2 * A_M * M_L - A_M 
          = A_M * (2 * M_L - 1)
        """

        mask_prediction = self.audionet_upconvlayer5(
            torch.cat((audio_upconv4feature, audio_conv1feature), dim=1), visual_feat
        ) * 2 - 1
        mask_prediction = self.interpolation(mask_prediction, x)

        return mask_prediction, ssl_loss
