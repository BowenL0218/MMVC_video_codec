import glob
import re
import os

import argparse
from model_auto_encoder import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import PIL
from VTL_data import VTL_dataset
from torchvision import transforms
from PIL import Image
import scipy.io as sc
import logging

torch.backends.cudnn.enabled = True
gpu_num = torch.cuda.device_count()
feature_channel = 96
latent_channel = 192

# pretrained_path = '/nfs/turbo/coe-hunseok/mrakeshc/vimeo_triplet_video_compression/deep_encoder_increased_latent_2048_temp_5_frames_sim_entr/deep_encoder_increased_latent_2048_temp_5_frames_sim_entr/iter_107730.pth.tar'
pretrained_path = '/nfs/turbo/coe-hunseok/mrakeshc/vimeo_triplet_video_compression/deep_encoder_auto_encoder_latent_512_512_3_64_64_96_no_res_no_sigmoid/iter_147212.pth.tar'
root_dir = '/nfs/turbo/coe-hunseok/mrakeshc/image_compression/gain_cvpr2021/images/'
text_name = 'Kodak_performance.txt'

logger = logging.getLogger("UVG_testing")


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]
def test():
    sumBpp = 0
    sumPsnr = 0
    sumMsssim = 0
    sumMsssimDB = 0
    cnt = 0
    residuals = []
    image_list = []
    root_dir_path = sorted(os.listdir(root_dir))
    for filename in sorted(glob.iglob(root_dir + '**/*.png', recursive=True)):
        image_list.append(filename)
        image_list.sort(key=natural_keys)
    video_len = len(image_list)
    for filename in image_list:
        prev = PIL.Image.open(filename).convert("RGB")
        
        torch_transform = transforms.Compose([
#                         transforms.CenterCrop((256,256)),
#                         transforms.CenterCrop((1024,1920)),
#                         transforms.CenterCrop((512, 512)),
                        transforms.ToTensor(),
                    ])
        prev = torch_transform(prev).unsqueeze(dim = 0)
#         print(prev.shape)
#         print(torch.max(prev))
        recon_cur = net(prev.to('cuda'))
#         print(torch.max(recon_cur))
#         print(torch.min(recon_cur))
        clipped_recon_image = recon_cur.clamp(0., 1.)
        mse_loss = torch.mean((clipped_recon_image - prev.to('cuda')).pow(2))
        psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
        sumPsnr += psnr
        cpu_recon_image = clipped_recon_image.cpu().detach()
        np_recon_image = cpu_recon_image.permute(0,2,3,1).squeeze().numpy()
        image = Image.fromarray((np_recon_image*255).astype(np.uint8))
        image.save('/nfs/turbo/coe-hunseok/mrakeshc/recon_kodak/recon_{}.png'.format(cnt+1))
        orig_image = prev.permute(0,2,3,1).squeeze().numpy()
        orig_image = Image.fromarray((orig_image*255).astype(np.uint8))
        orig_image.save('/nfs/turbo/coe-hunseok/mrakeshc/recon_kodak/orig_{}.png'.format(cnt+1))
        msssim = ms_ssim(cpu_recon_image, prev, data_range=1.0, size_average=True)
        msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
        sumMsssimDB += msssimDB
        sumMsssim += msssim
        cnt += 1
        logger.info("Num: {}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(cnt, psnr, msssim, msssimDB))
        print("Num: {}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(cnt, psnr, msssim, msssimDB))
        if cnt == 17:
            break

#     logger.info("Test on Vimeo triplet dataset: model-{}".format(step))
    sumPsnr /= cnt
    sumMsssim /= cnt
    sumMsssimDB /= cnt
    logger.info("Dataset Average result---Dataset Num: {}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(cnt, sumPsnr, sumMsssim, sumMsssimDB))
    print("Dataset Average result---Dataset Num: {}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(cnt, sumPsnr, sumMsssim, sumMsssimDB))


if __name__ == "__main__":
    save_path = '/home/mrakeshc/RAFT_video_compression/'
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.ERROR)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    tb_logger = None
    filehandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
    filehandler.setLevel(logging.INFO)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    logger.info("Testing")
    model = VideoCoder(1, feature_channel, latent_channel)
#     filename_list = []
#     for filename in glob.glob('/nfs/turbo/coe-hunseok/mrakeshc/vimeo_triplet_video_compression/deep_encoder_increased_latent_2048_temp_5_frames_sim_entr_noise_round_2_bit_0.25/*.tar'): #assuming gif
#         filename_list.append(filename)
#     for filename in glob.glob('/nfs/turbo/coe-hunseok/mrakeshc/vimeo_triplet_video_compression/deep_encoder_increased_latent_2048_temp_5_frames_sim_entr_noise_round_2_bit_0.25/*/*.tar'): #assuming gif
#         filename_list.append(filename)
#     for i in range(len(filename_list)):
#         pre_trained_path = filename_list[i]
#         with open('Per_video_performance_VTL_sim_entr_noise_round_2_bit_0.25_multiple.txt', 'a') as f_text:
#             f_text.write(pre_trained_path)
#             f_text.write("\n")
#         load_model(model, pre_trained_path)
#         net = model.cuda()
#         test()
    load_model(model, pretrained_path)
    net = model.cuda()
    test()
