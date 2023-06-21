import torch
import torch.nn as nn
import torch.nn.functional as F
from .GDN import GDN
import math



class ResBlock(nn.Module):
    def __init__(
        self, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feat, n_feat, kernel_size, bias=bias, stride =1, padding = 1))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

    
class ResGDN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, inv=False):
        super(ResGDN, self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.inv = bool(inv)
        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch,
                               self.k, self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.in_ch, self.out_ch,
                               self.k, self.stride, self.padding)
        self.ac1 = GDN(self.in_ch, self.inv)
        self.ac2 = GDN(self.in_ch, self.inv)

    def forward(self, x):
        x1 = self.ac1(self.conv1(x))
        x2 = self.conv2(x1)
        out = self.ac2(x + x2)
        return out



class ResBlock_(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ResBlock_, self).__init__()
        self.in_ch = int(in_channel)
        self.out_ch = int(out_channel)
        self.k = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch,
                                self.k, self.stride, self.padding)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.conv2 = nn.Conv2d(self.in_ch, self.out_ch,
                                self.k, self.stride, self.padding)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)

    def forward(self, x):
        x1 = self.conv2(F.relu(self.conv1(x)))
        out = x+x1
        return out


# class NonLocalBlock2D(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(NonLocalBlock2D, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.g = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
#         self.theta = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
#         self.phi = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
#         self.W = nn.Conv2d(self.out_channel, self.in_channel, 1, 1, 0)
#         nn.init.constant(self.W.weight, 0)
#         nn.init.constant(self.W.bias, 0)

#     def forward(self, x):
#         # x_size: (b c h w)

#         batch_size = x.size(0)
#         g_x = self.g(x).view(batch_size, self.out_channel, -1)
#         g_x = g_x.permute(0, 2, 1)
#         theta_x = self.theta(x).view(batch_size, self.out_channel, -1)
#         theta_x = theta_x.permute(0, 2, 1)
#         phi_x = self.phi(x).view(batch_size, self.out_channel, -1)

#         f1 = torch.matmul(theta_x, phi_x)
#         f_div_C = F.softmax(f1, dim=-1)
#         y = torch.matmul(f_div_C, g_x)
#         y = y.permute(0, 2, 1).contiguous()
#         y = y.view(batch_size, self.out_channel, *x.size()[2:])
#         W_y = self.W(y)
#         z = W_y+x

#         return z


class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)
        
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        
        g_x = g_x.permute(0,2,1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        
        theta_x = theta_x.permute(0,2,1)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
       
        f_div_C = F.softmax(f, dim=1)
        
        
        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0,2,1).contiguous()
         
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z
