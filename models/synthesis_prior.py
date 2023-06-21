import math
import torch.nn as nn
import torch

class Synthesis_prior_net(nn.Module):
    '''
    Decode synthesis prior
    '''
    def __init__(self, in_channel=192, out_channel=384):
#     def __init__(self, in_channel=192, out_channel=192):
        super(Synthesis_prior_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channel, in_channel, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1)))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(in_channel, 288, 5, stride=2, padding=2, output_padding=1)
#         self.deconv2 = nn.ConvTranspose2d(in_channel, 288, 3, stride=1, padding=1, output_padding=0)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, (math.sqrt(2 * 1 * (in_channel + 288) / (2 * in_channel))))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(288, out_channel, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2 * 1 * (288 + out_channel) / (288 * 2))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        return self.deconv3(x)
