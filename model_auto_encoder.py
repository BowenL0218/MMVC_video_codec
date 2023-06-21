import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from models import *
import matplotlib.pyplot as plt



def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0

    
class VideoCoder(nn.Module):
    def __init__(self, batch_size, feature_channel=192, latent_channel=384):
        super(VideoCoder, self).__init__()
        self.Encoder = Feature_encoder(feature_channel)
        self.Decoder = Feature_decoder(in_channel=feature_channel, mid_channel=feature_channel)

        self.feature_channel = feature_channel
        self.latent_channel = latent_channel
    
    def forward(self, prev):
        prev_feature = self.Encoder(prev)
        batch_size = prev_feature.size()[0]
        recon_cur = self.Decoder(prev_feature)
        return recon_cur