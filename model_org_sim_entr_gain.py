import numpy as np
import os
import torch
import torchvision.models as models
# from torch.autograd import Variable
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
from RAFT_test import demo
import matplotlib.pyplot as plt


def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))

def load_model_rev(model, f, name):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        new_dict = {}
#         for k, _ in pretrained_dict.items():
#             print("this keys are in the original model", k)
        for k, _ in model_dict.items():
#             print("this keys are in the combined model", k)
            if (k == "deconv1.weight"):
                new_dict[k] = pretrained_dict[name + k][:192,:,:,:]
            else:
#                 print("This are the keys in the Encoder and Decoder:", k)
                new_dict[k] = pretrained_dict[name + k]
#             print(name + k, pretrained_dict[name + k].size())
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
# path = '/scratch/hunseok_root/hunseok0/mrakeshc/flickr_dataset/checkpoint/with_attention_rev_32/iter_979751.pth.tar' 

def load_model_deep(model, f, name):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        new_dict = {}
#         for k, _ in pretrained_dict.items():
#             print("this keys are in the original model", k)
        for k, _ in model_dict.items():
#             print("this keys are in the combined model", k)
            if (k == "deconv1.weight"):
                new_dict[k] = pretrained_dict[name + k][:192,:,:,:]
            else:
#                 print("The weights are being loaded")
#                 print("This are the keys in the Encoder and Decoder:", k)
                new_dict[k] = pretrained_dict[name + k]
#             print(name + k, pretrained_dict[name + k].size())
#         smaple
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)

path = '/home/mrakeshc/NIC/code/output/msssim4.pth.tar'
# path = '/home/mrakeshc/NIC/code/output_1/mse400.pth.tar'

def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        new_dict = {}
#         print(model_dict.keys())
#         for k, v in pretrained_dict.items():
#             if (k in model_dict) and (k != "Encoder.conv1.weight") and (k != "Encoder.conv1.bias"):
# #                 print("these are the keys:",k)
#                 new_dict[k] = v
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
#         model_dict.update(new_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0

class VideoCoder(nn.Module):
    def __init__(self, feature_channel=192, latent_channel=384):
        super(VideoCoder, self).__init__()
        self.Encoder = Feature_encoder(feature_channel)
        self.Decoder = Feature_decoder(in_channel=feature_channel, mid_channel=feature_channel)
        self.temporalPredictor = Temporal_predictor(in_channel=feature_channel*5, out_channel=feature_channel)

        self.priorEncoder = Analysis_prior_net(in_channel=feature_channel, out_channel=feature_channel)
        self.priorDecoder = Synthesis_prior_net(in_channel=feature_channel, out_channel=feature_channel)
        self.bitEstimator_z = Bit_estimator(channel=feature_channel)
        self.contextModel = Context_model_autoregressive(in_channel=feature_channel, out_channel=latent_channel)
        self.entropyParameters = Entropy_parameters(in_channel=latent_channel*2, out_channel=latent_channel)

        self.feature_channel = feature_channel
        self.latent_channel = latent_channel
    
#     def forward(self, prev1, prev2, cur):
    def forward(self, prev1, prev2, prev3, prev4, cur, residuals_list, dir_image, diff, num_unchanged_total, num_total_blocks, index):
        quant_noise_feature = torch.zeros(cur.size(0), self.feature_channel, cur.size(2) // 4, cur.size(3) // 4).cuda()

        quant_noise_z = torch.zeros(cur.size(0), self.feature_channel, cur.size(2) // 8, cur.size(3) // 8).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
        latent_diff = torch.zeros(cur.size(0), self.feature_channel, cur.size(2) // 8, cur.size(3) // 8).cuda()

        prev1_feature = self.Encoder(prev1)
        prev2_feature = self.Encoder(prev2)
        prev3_feature = self.Encoder(prev3)
        prev4_feature = self.Encoder(prev4)
        cur_feature = self.Encoder(cur)
        warpped_image = demo(prev3, prev4)
        warpped_features = self.Encoder(warpped_image)


        batch_size = prev2_feature.size()[0]
        cur_pred = self.temporalPredictor(prev1_feature, prev2_feature, prev3_feature, prev4_feature, warpped_features)

        
        residual = cur_feature - cur_pred
        residual[:,index,:,:] = 0
        # residual *= latent_diff
        # residual = latent_diff
        # reisdual = torch.mul(residual, latent_diff)
        

        
        z = self.priorEncoder(residual)
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)
            # compressed_z = torch.round(z*2)/2
        hd_out = self.priorDecoder(compressed_z)
        
        feature_renorm  = residual
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        residuals_list.append(torch.squeeze(residual.cpu()).numpy())
        recon_cur = self.Decoder(cur_pred+compressed_feature_renorm)
        recon_sigma = hd_out
        def feature_probs_based_sigma(feature, sigma):
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-10, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob
        total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        im_shape = cur.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z
        clipped_recon_cur = recon_cur.clamp(0., 1.)
        mse_loss = torch.mean((clipped_recon_cur - cur).pow(2))
        return clipped_recon_cur, mse_loss, bpp_feature, bpp_z, bpp
