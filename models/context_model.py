import math
import torch.nn as nn
import torch.nn.functional as F
import torch

class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        """
        Masked convolutional layer. 

        Input: 
        - mask_type: "A" or "B" 
        """
        assert mask_type == 'A' or mask_type == 'B'
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)

    def forward(self, input, cond=None):
        batch_size = input.shape[0]
        out = F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)
        return out
  
    def create_mask(self, mask_type):
        _, _, h, w = self.mask.shape
        # if mask_type == 'A': 
        self.mask[:,:,:h//2,:] = 1
        self.mask[:,:,h//2,:w//2] = 1
        if mask_type == 'B':
            self.mask[:,:,h//2,w//2] = 1


class Context_model_autoregressive(nn.Module):
    '''
    Context model
    '''
    def __init__(self, in_channel=192, out_channel=384):
        super(Context_model_autoregressive, self).__init__()
        self.masked_conv_3 = MaskConv2d('A', in_channel, out_channel, kernel_size = 3, padding = 1)
        self.masked_conv_5 = MaskConv2d('A', in_channel, out_channel, kernel_size = 5, padding = 2)
        self.masked_conv_7 = MaskConv2d('A', in_channel, out_channel, kernel_size = 7, padding = 3)

    def forward(self, x):
        # out = torch.cat((self.masked_conv_3(x), self.masked_conv_5(x), self.masked_conv_7(x)), dim=1)
       
        return self.masked_conv_5(x)

