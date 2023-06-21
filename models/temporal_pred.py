import math
import torch
import torch.nn as nn

class Temporal_predictor(nn.Module):
    def __init__(self, in_channel=192*3, out_channel=192):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, 512, 1, stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (in_channel + 512)/(2 * in_channel))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(512, 256, 1, stride =1, padding=0)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, (math.sqrt(2 * (512 + 256)/(2 * 512))))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(256, out_channel, 1, stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, (math.sqrt(2 * (256 + out_channel)/(2 * 256))))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

#     def forward(self, x_prev1, x_prev2):
#         x = torch.cat((x_prev1, x_prev2), dim=1)
#     def forward(self, x_prev1, x_prev2, pred):
    def forward(self, x_prev1, x_prev2, x_prev3, x_prev4, pred):
        # batch_size = x_prev1.shape[0]
        # pred = pred.repeat(batch_size, 1, 1, 1)
#         x = torch.cat((x_prev1, x_prev2, pred), dim=1)
        x = torch.cat((x_prev1, x_prev2, x_prev3, x_prev4, pred), dim=1)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)