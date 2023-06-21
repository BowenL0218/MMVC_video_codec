import torch
import math
import torch.nn as nn
from .GDN import GDN

class Feature_encoder(nn.Module):
    '''
    Encodes video frames to feature domain.
    '''
    def __init__(self, latent_channel):
        super(Feature_encoder, self).__init__()
        self.conv_layer1 = self.make_layers(3, [32, 32, 64, 64])
        self.conv_layer2 = self.make_layers(128, [128, 128, 256, 256])
        self.conv_layer3 = self.make_layers(512, [512, 512, 512, 512])
        
        self.conv1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(128)
        self.conv2 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(512)
        self.conv3 = nn.Conv2d(512, latent_channel, 3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(512, latent_channel, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(latent_channel)

        
    def make_layers(self, in_channels, stem_channels):
        stem_conv = []
        for out_channels in stem_channels:
            m = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
            torch.nn.init.xavier_normal_(m.weight.data, (math.sqrt(2)))
            torch.nn.init.constant_(m.bias.data, 0.01)
            stem_conv.append(m)
            stem_conv.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        stem_conv = nn.Sequential(*stem_conv)
        return stem_conv
        

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.gdn1(self.conv1(x))
        x = self.conv_layer2(x)
        x = self.gdn2(self.conv2(x))
        x = self.conv_layer3(x)
        x = self.gdn3(self.conv3(x))
        return x

# from torchsummary import summary
# model = Feature_encoder(192)
# model.cuda()
# summary(model, (3, 256, 256))



