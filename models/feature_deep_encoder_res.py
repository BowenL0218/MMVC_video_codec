import torch
import math
import torch.nn as nn
from .GDN import GDN
import torch.nn.functional as F


# class ResBlock(nn.Module):
#     def __init__(
#         self, inp_feat, n_feat, kernel_size,
#         bias=True, bn=True, act=nn.ReLU(True), res_scale=1):

#         super(ResBlock, self).__init__()
#         m = []
#         for i in range(2):
#             n = nn.Conv2d(inp_feat, n_feat, kernel_size, bias=bias, stride =1, padding = 1)
#             torch.nn.init.xavier_normal_(n.weight.data, (math.sqrt(2)))
#             torch.nn.init.constant_(n.bias.data, 0.01)
#             m.append(n)
#             if bn: m.append(nn.BatchNorm2d(n_feat))
#             if i == 0: m.append(act)
#             inp_feat = n_feat

#         self.body = nn.Sequential(*m)
#         self.res_scale = res_scale

#     def forward(self, x):
#         res = self.body(x).mul(self.res_scale)
#         res += x
#         return F.relu(res)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 3, bias = True, stride = 1, padding = 1):
        super(ResBlock, self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch,
                                self.k, bias = bias, stride = self.stride, padding = self.padding)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.bn1 = nn.BatchNorm2d(self.out_ch)
        self.conv2 = nn.Conv2d(self.in_ch, self.out_ch,
                                self.k, bias = bias, stride = self.stride, padding = self.padding)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.bn2 = nn.BatchNorm2d(self.out_ch)

    def forward(self, x):
        x1 = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        out = x+x1
        out = F.relu(out)
        return out


class Feature_encoder(nn.Module):
    '''
    Encodes video frames to feature domain.
    '''
    def __init__(self, latent_channel):
        super(Feature_encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(32)
        self.res_block1 = ResBlock(32, 32, 3)
        self.res_block2 = ResBlock(32, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(64)
        self.res_block3 = ResBlock(64, 64, 3)
        self.res_block4 = ResBlock(64, 64, 3)
        self.conv3 = nn.Conv2d(64, latent_channel, 3, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(64, latent_channel, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(latent_channel)
        self.res_block5 = ResBlock(latent_channel, latent_channel, 3)
        self.res_block6 = ResBlock(latent_channel, latent_channel, 3)
        

                

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.gdn2(self.conv2(x))
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.gdn3(self.conv3(x))
        x = self.res_block5(x)
        x = self.res_block6(x)
        return x

# from torchsummary import summary
# model = Feature_encoder(96)
# print(model)
# model.cuda()
# summary(model, (3, 512, 512))
