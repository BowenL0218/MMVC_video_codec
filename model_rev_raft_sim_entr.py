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
from RAFT_test import demo
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

# path = '/scratch/hunseok_root/hunseok0/mrakeshc/flickr_dataset/checkpoint/with_attention_rev_32/iter_979751.pth.tar'
path = '/home/mrakeshc/NIC/code/output_1/mse400.pth.tar'

    
class VideoCoder(nn.Module):
    def __init__(self, batch_size, feature_channel=192, latent_channel=384):
        super(VideoCoder, self).__init__()
#         feature_channel = 192
        self.Encoder = Feature_encoder(feature_channel)
        self.Decoder = Feature_decoder(in_channel=feature_channel, mid_channel=feature_channel)
        # PixelCNN spatial predictor
#         self.spatialPredictor = Spatial_predictor(input_shape=(192, 16, 16), n_filters=384, kernel_size=5, n_layers=7)
#         self.spatialPredictor = Spatial_predictor(input_shape=(192, 32, 32), n_filters=384, kernel_size=5, n_layers=7)
#         self.temporalPredictor = Temporal_predictor(in_channel=feature_channel*3, out_channel=feature_channel)
        self.temporalPredictor = Temporal_predictor(in_channel=feature_channel*5, out_channel=feature_channel)
#         self.temporalPredictor = ConvLSTM(feature_channel, [feature_channel]*15, (3, 3), 15, True, True, False)
#         self.temporalPredictor = Temporal_predictor(in_channel=feature_channel*2, out_channel=feature_channel)
        self.priorEncoder = Analysis_prior_net(in_channel=feature_channel, out_channel=feature_channel)
#         self.priorDecoder = Synthesis_prior_net(in_channel=feature_channel, out_channel=latent_channel)
        self.priorDecoder = Synthesis_prior_net(in_channel=feature_channel, out_channel=feature_channel)
        self.bitEstimator_z = Bit_estimator(channel=feature_channel)
        self.contextModel = Context_model_autoregressive(in_channel=feature_channel, out_channel=latent_channel)
        self.entropyParameters = Entropy_parameters(in_channel=latent_channel*2, out_channel=latent_channel)
        self.rand_num = nn.parameter.Parameter(torch.tensor(4.0), requires_grad=True)
        # self.rand_vectors = nn.parameter.Parameter(torch.randn(1, feature_channel, 32, 32), requires_grad=True)

        self.feature_channel = feature_channel
        self.latent_channel = latent_channel
    
#     def forward(self, prev1, prev2, cur):
    def forward(self, prev1, prev2, prev3, prev4, cur):
#         quant_noise_feature = torch.zeros(cur.size(0), self.feature_channel, cur.size(2) // 16, cur.size(3) // 16).cuda()
        quant_noise_feature = torch.zeros(cur.size(0), self.feature_channel, cur.size(2) // 8, cur.size(3) // 8).cuda()
#         quant_noise_feature = torch.zeros(cur.size(0), self.feature_channel, cur.size(2) // 4, cur.size(3) // 4).cuda()
#         quant_noise_z = torch.zeros(cur.size(0), self.feature_channel, cur.size(2) // 64, cur.size(3) // 64).cuda()
        quant_noise_z = torch.zeros(cur.size(0), self.feature_channel, cur.size(2) // 32, cur.size(3) // 32).cuda()
#         quant_noise_z = torch.zeros(cur.size(0), self.feature_channel, cur.size(2) // 8, cur.size(3) // 8).cuda()
#         rand_num = nn.parameter.Parameter(torch.tensor(4.0), requires_grad=True)
        rand_ = torch.clone(self.rand_num).detach()
#         print(self.rand_num)
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
#         quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5/rand_.cpu().numpy(), 0.5/rand_.cpu().numpy())
#         quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5/rand_.cpu().numpy(), 0.5/rand_.cpu().numpy())


        prev1_feature = self.Encoder(prev1)
        prev2_feature = self.Encoder(prev2)
        prev3_feature = self.Encoder(prev3)
        prev4_feature = self.Encoder(prev4)
        cur_feature = self.Encoder(cur)
        warpped_image = demo(prev3, prev4)
        # out = warpped_image[0].cpu().permute(1,2,0)
        # out = np.float32(out)
        # plt.imsave("image.png", out)
        # out1 = cur[0].cpu().permute(1,2,0)
        # out1 = np.float32(out1)
        # plt.imsave("image1.png", out1)
        # out2 = prev1[0].cpu().permute(1,2,0)
        # out2 = np.float32(out2)
        # plt.imsave("imag2.png", out2)
        # out3 = prev2[0].cpu().permute(1,2,0)
        # out3 = np.float32(out3)
        # plt.imsave("image3.png", out3)
        warpped_features = self.Encoder(warpped_image)
#         warpped_features_cpu = warpped_features[0].cpu().permute(1,2,0)
#         warpped_features_float = np.float32(warpped_features_cpu)
#         plt.imsave("image3.png", warpped_features_float)
#         print(cur_feature.shape)
        batch_size = cur_feature.size()[0]

        # Spatial and temporal prediction
#         cur_pred = self.spatialPredictor(cur_feature)
#         cur_pred = self.temporalPredictor(prev1_feature, prev2_feature, cur_pred)
        # cur_pred = self.temporalPredictor(prev1_feature, prev2_feature, self.rand_vectors)
#         cur_pred = self.temporalPredictor(prev1_feature, prev2_feature, warpped_features)
        cur_pred = self.temporalPredictor(prev1_feature, prev2_feature, prev3_feature, prev4_feature, warpped_features)
#         cur_pred = torch.cat((prev1_feature.unsqueeze(axis = 1), prev2_feature.unsqueeze(axis = 1), prev3_feature.unsqueeze(axis = 1), prev4_feature.unsqueeze(axis = 1), warpped_features.unsqueeze(axis = 1)), dim=1)
#         _, cur_pred = self.temporalPredictor(cur_pred)
#         cur_pred = cur_pred[0][0]
#         cur_pred = self.temporalPredictor(prev1_feature, prev2_feature)
        
        # residual is the elementwise difference between cur_feature and the prediction
        residual = cur_feature - cur_pred
        
        z = self.priorEncoder(residual)
#         print(z.shape)
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)
#             compressed_z = torch.round(z*2)/2
        hd_out = self.priorDecoder(compressed_z)
        
        feature_renorm  = residual
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
#             compressed_feature_renorm = torch.round(feature_renorm*2)/2
        recon_cur = self.Decoder(cur_pred+compressed_feature_renorm)
        recon_sigma = hd_out
        def feature_probs_based_sigma(feature, sigma):
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-10, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            return probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            return prob

        probs_features = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        prob_z = iclr18_estimate_bits_z(compressed_z)
        recon_cur = recon_cur + torch.tensor(0.01).cuda()
#         return clipped_recon_cur, mse_loss, bpp_feature, bpp_z, bpp
#         return clipped_recon_cur, compressed_feature_renorm, recon_sigma, recon_mu, compressed_z
        return recon_cur, probs_features, prob_z
