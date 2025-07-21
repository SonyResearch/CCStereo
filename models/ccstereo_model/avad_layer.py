import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class AVAD(nn.Module):
    def __init__(self, norm_nc, audio_size, vision_channel=512, v_seq=98, ks=5):
        super().__init__()
        pw = ks // 2
        nhidden = norm_nc
        label_nc = v_seq
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.pos_embed_v = nn.Parameter(torch.randn(1, v_seq, norm_nc) * .02)
        self.pos_embed_a = nn.Parameter(torch.randn(1, audio_size[0]*audio_size[1], norm_nc) * .02)
        self.attn_layer = torch.nn.MultiheadAttention(embed_dim=nhidden, num_heads=8, batch_first=True)
        self.visual_conv = nn.Conv2d(vision_channel, norm_nc, kernel_size=1)
        self.flow_conv = nn.Conv2d(vision_channel, norm_nc, kernel_size=1)

    def forward(self, x, v_fea, flow_feat=None):
        _, _, ah, aw = x.shape
        v_fea = self.visual_conv(v_fea)
        normalized = self.param_free_norm(x)
        a_fea = normalized.clone()
        a_fea = rearrange(a_fea, 'b c h w -> b (h w) c')
        v_fea = rearrange(v_fea, 'b c h w -> b (h w) c')
        a_fea = a_fea + self.pos_embed_a
        v_fea = v_fea + self.pos_embed_v
        cross_a_fea = F.normalize(a_fea, p=2, dim=-1)
        cross_v_fea = F.normalize(v_fea, p=2, dim=-1)
        cross_attn_fea = torch.bmm(cross_v_fea, cross_a_fea.permute(0, 2, 1))
        cross_attn_fea = rearrange(cross_attn_fea, 'b c (h w) -> b c h w', h=ah, w=aw)
        actv = self.mlp_shared(cross_attn_fea)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta

        return out


