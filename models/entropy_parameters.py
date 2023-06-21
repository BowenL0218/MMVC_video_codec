import math
import torch.nn as nn
import torch.nn.functional as F
import torch

class Entropy_parameters(nn.Module):
    def __init__(self, in_channel=384*2, out_channel=384):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, 640, 1, stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (in_channel + 640) / (2 * in_channel))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(640, 512, 1, stride =1, padding=0)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, (math.sqrt(2 * (640 + 512) / (2 * 640))))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(512, out_channel, 1, stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, (math.sqrt(2 * (512 + out_channel) / (2 * 512))))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.out_channel = out_channel

    def forward(self, cm_out, hd_out):
        x = torch.cat((cm_out, hd_out), dim=1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return torch.exp(x[:, :self.out_channel//2, :, :]), x[:, self.out_channel//2:, :, :]