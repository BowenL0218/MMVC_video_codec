import math
import torch.nn as nn
import torch
from .GDN import GDN
import torch.nn.functional as F


class Feature_decoder(nn.Module):
    '''
    Decodes feature domain video frames to pixel domain.
    '''
    def __init__(self, in_channel, mid_channel):
        super(Feature_decoder, self).__init__()
        self.deconv_layer1 = self.make_layers(512, [512, 512, 512, 512])
        self.deconv_layer2 = self.make_layers(256, [256, 128, 128, 128])
        self.deconv_layer3 = self.make_layers(64, [64, 32, 32, 3])
        self.igdn1 = GDN(mid_channel, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(mid_channel, 512, 3, stride=2, padding=1, output_padding=1)
#         self.deconv2 = nn.ConvTranspose2d(mid_channel, 512, 3, stride=1, padding=1, output_padding=0)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(512, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(128, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * 1 * (3 + 64) / (2 * 64))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        
    def make_layers(self, in_channels, stem_channels):
        stem_conv = []
        for out_channels in stem_channels:
            stem_conv.append(nn.ReLU(inplace=True))
            m = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=1, padding=1, output_padding=0)
            torch.nn.init.xavier_normal_(m.weight.data, (math.sqrt(2)))
            torch.nn.init.constant_(m.bias.data, 0.01)
            stem_conv.append(m)
            in_channels = out_channels

        stem_conv = nn.Sequential(*stem_conv)
        return stem_conv
        

    def forward(self, x):
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.deconv_layer1(x)
        x = self.deconv3(self.igdn2(x))
        x = self.deconv_layer2(x)
        x = self.deconv4(self.igdn3(x))
        x = self.deconv_layer3(x)
        # x = F.sigmoid(x)
        return x

# from torchsummary import summary
# model = Feature_decoder(192, 192)
# model.cuda()
# summary(model, (192, 32, 32))
