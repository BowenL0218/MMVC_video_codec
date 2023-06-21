import math
import torch
import torch.nn as nn

class Analysis_prior_net(nn.Module):
    '''
    Analysis prior net
    '''
    def __init__(self, in_channel, out_channel):
        super(Analysis_prior_net, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (in_channel + out_channel) / (2 * in_channel))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2)
#         self.conv2 = nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channel, out_channel, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
    
    def forward(self, x):
        x = torch.abs(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)

    