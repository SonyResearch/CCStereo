import torch
import torch.nn as nn
from loguru import logger

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d, stride=1):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=stride, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downrelu, downconv, downnorm])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d, up=False):
    uprelu = nn.ReLU(True)
    upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if up else nn.Identity()
    upconv = nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1)
    # upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1)
    upnorm = norm_layer(output_nc)
    return nn.Sequential(*[uprelu, upsample, upconv, upnorm])


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
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)



class AudioModel(torch.nn.Module):
    def __init__(self, backbone, pretrain_path, out_plane, num_classes=2):
        super(AudioModel, self).__init__()
        self.backbone = AudioVisual5layerUNet()
        self.cls_head = nn.Linear(out_plane, num_classes)
        # self.cls_head = nn.Linear(out_plane, 1)

    def forward(self, x):
        return self.backbone(x)

    def load_audio_model(self, path_, strict=True):
        # return
        param_dict = torch.load(path_)
        param_ = self.backbone.state_dict()
        out_, in_ = param_["embeddings.4.weight"].shape
        param_dict["embeddings.4.weight"] = torch.nn.init.kaiming_normal_(
            torch.zeros(out_, in_)
        )
        param_dict["embeddings.4.bias"] = torch.zeros(out_)
        self.backbone.load_state_dict(param_dict, strict=strict)


def make_layers():
    layers = []
    in_channels = 1
    # profile = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]
    profile = [64, "M", 128, "M", 256, 256]
    for v in profile:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class AudioVisual5layerUNet(nn.Module):
    def __init__(self, ngf=32, input_nc=1, output_nc=256):
        super(AudioVisual5layerUNet, self).__init__()

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf, stride=2)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2, stride=2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4, stride=2)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2, up=True)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf, up=True)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, up=True) #outermost layer use a sigmoid to bound the mask

    def forward(self, x):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)

        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_embed = audio_conv5feature
        audio_upconv1feature = self.audionet_upconvlayer1(audio_conv5feature)

        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))
        return audio_embed, mask_prediction