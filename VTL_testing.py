import glob
import re
import os

import argparse
# from model_org import *
from model_org_sim_entr import *
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
feature_channel = 192
latent_channel = 384

# pretrained_path  = '/nfs/turbo/coe-hunseok/mrakeshc/vimeo_triplet_video_compression/deep_encoder_increased_latent_2048_temp_5_frames_sim_entr_0.125_noise_round_4/iter_32312.pth.tar'
pretrained_path = '/nfs/turbo/coe-hunseok/mrakeshc/vimeo_triplet_video_compression/deep_encoder_increased_latent_2048_temp_5_frames_sim_entr/deep_encoder_increased_latent_2048_temp_5_frames_sim_entr/iter_107730.pth.tar'
# pretrained_path = '/nfs/turbo/coe-hunseok/mrakeshc/vimeo_triplet_video_compression/deep_encoder_increased_latent_2048_temp_5_frames_RAFT_simp_entr_noise_round_0_bit_0.5_192_channel_sigmoid_v2/iter_193840.pth.tar'
# pretrained_path  = '//nfs/turbo/coe-hunseok/mrakeshc/vimeo_triplet_video_compression/deep_encoder_increased_latent_2048_temp_5_frames_sim_entr_noise_round_2_bit_0.25/iter_32312.pth.tar'
# pretrained_path = '/nfs/turbo/coe-hunseok/mrakeshc/vimeo_triplet_video_compression/deep_encoder_increased_latent_2048_temp_5_frames_simp_entr_noise_round_0_bit_0.5/iter_55392.pth.tar'
# pretrained_path = '/nfs/turbo/coe-hunseok/mrakeshc/vimeo_triplet_video_compression/deep_encoder_increased_latent_2048_temp_5_frames_comp_entr_noise_round_0_bit_0.5/iter_48468.pth.tar'
# pretrained_path = '/nfs/turbo/coe-hunseok/mrakeshc/vimeo_triplet_video_compression/deep_encoder_increased_latent_2048_temp_5_frames_comp_entr_noise_round_0_bit_0.5_24_channel/iter_48468.pth.tar'
# pretrained_path = '/nfs/turbo/coe-hunseok/mrakeshc/vimeo_triplet_video_compression/deep_encoder_increased_latent_0.9_temp_5_frames_simp_entr_noise_round_0_bit_0.5/iter_55392.pth.tar'
# pretrained_path = '/nfs/turbo/coe-hunseok/mrakeshc/vimeo_triplet_video_compression/deep_encoder_increased_latent_2048_temp_5_frames_comp_entr_noise_round_0_bit_0.5_16_channel/iter_46160.pth.tar'
# root_dir = '/nfs/turbo/coe-hunseok/mrakeshc/VTL_dataset/'
root_dir = '/nfs/turbo/coe-hunseok/mrakeshc/UVG_dataset/'
text_name = 'Per_video_performance_VTLdeep_encoder_increased_latent_2048_temp_5_frames_sim_entr_del_lat.txt'

logger = logging.getLogger("UVG_testing")

# def force_cudnn_initialization():
#     s = 32
#     dev = torch.device('cuda')
#     torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
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
#     force_cudnn_initialization()
    sumBpp = 0
    sumPsnr = 0
    sumMsssim = 0
    sumMsssimDB = 0
    cnt = 0
    residuals = []
    root_dir_path = sorted(os.listdir(root_dir))
    print(root_dir_path)
#     for dir_name in root_dir_path:
    for i in range(1):
        dir_name = root_dir_path[4]
        image_list = []
        folder_path = root_dir + dir_name + '/'
        for filename in sorted(glob.iglob(folder_path + '**/*.png', recursive=True)):
            image_list.append(filename)
        image_list.sort(key=natural_keys)
        video_len = len(image_list)
        # print(image_list)

        prev1 = PIL.Image.open(image_list[0]).convert("RGB")
        prev2 = PIL.Image.open(image_list[1]).convert("RGB")
        prev3 = PIL.Image.open(image_list[2]).convert("RGB")
        prev4 = PIL.Image.open(image_list[3]).convert("RGB")
        torch_transform = transforms.Compose([
#                         transforms.CenterCrop((256,256)),
                        transforms.CenterCrop((1024,1920)),
                        transforms.ToTensor(),
                    ])
        prev1 = torch_transform(prev1).unsqueeze(dim = 0)
        prev2 = torch_transform(prev2).unsqueeze(dim = 0)
        prev3 = torch_transform(prev3).unsqueeze(dim = 0)
        prev4 = torch_transform(prev4).unsqueeze(dim = 0)

        with torch.no_grad():
            num = 0
            sum_per_BPP = 0
            sum_per_psnr = 0
            sum_per_mssim = 0
            sum_per_mssimdb = 0
            print("Currently the video compression is done for", dir_name)
            net.eval()
            incr = 0
#             net.train()
            for cur_idx in image_list[4:]:
                incr += 1
                dir_image = '/nfs/turbo/coe-hunseok/mrakeshc/Images_reconstructed_inter/' + dir_name
                os.makedirs(dir_image, exist_ok=True)
                dir_image +=  '/VTL_' + str(incr)
                cur = PIL.Image.open(cur_idx).convert("RGB")
                cur = torch_transform(cur).unsqueeze(dim = 0)
    #             print(cur.shape)
                clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = net(prev1.to('cuda'), prev2.to('cuda'), prev3.to('cuda'), prev4.to('cuda'), cur.to('cuda'), residuals, dir_image)
#                 clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = net(prev1.to('cuda'), prev2.to('cuda'), prev3.to('cuda'), prev4.to('cuda'), cur.to('cuda'), residuals)
                mse_loss, bpp_feature, bpp_z, bpp = \
                    torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
                psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
                sumBpp += bpp
                sumPsnr += psnr
                cpu_recon_image = clipped_recon_image.cpu().detach()

                np_recon_image = cpu_recon_image.permute(0,2,3,1).squeeze().numpy()
                image = Image.fromarray((np_recon_image*255).astype(np.uint8))
                save_path = '/nfs/turbo/coe-hunseok/mrakeshc/reconstructed_images_train/recon_UVG_sim_entr/' + dir_name + '/'
                os.makedirs(save_path, exist_ok=True)
                image.save(save_path + '/recon_{}.png'.format(num+1))
                orig_image = cur.permute(0,2,3,1).squeeze().numpy()
                orig_image = Image.fromarray((orig_image*255).astype(np.uint8))
                orig_image.save(save_path + '/orig_{}.png'.format(num+1))
                msssim = ms_ssim(cpu_recon_image, cur, data_range=1.0, size_average=True)
                msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
                sumMsssimDB += msssimDB
                sumMsssim += msssim
                num += 1
                cnt += 1
                sum_per_BPP += bpp
                sum_per_psnr += psnr
                sum_per_mssim += msssim
                sum_per_mssimdb += msssimDB
                logger.info("Num: {}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(cnt, bpp, psnr, msssim, msssimDB))
                print("Num: {}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(cnt, bpp, psnr, msssim, msssimDB))
                with open(text_name, 'a') as f_text:
                    f_text.write("Num: {}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(cnt, bpp, psnr, msssim, msssimDB))
                    f_text.write("\n")
                prev1 = prev2
                prev2 = prev3
                prev3 = prev4
                prev4 = clipped_recon_image
                if num == 10:
                    break
            else:
                continue
            break
            with open(text_name, 'a') as f_text:
                f_text.write("Num: {}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(num, sum_per_BPP/num, sum_per_psnr/num, sum_per_mssim/num, sum_per_mssimdb/num))
                f_text.write("\n")

#                 prev1 = prev2
#                 prev2 = clipped_recon_image
#                 prev1 = prev2
#                 prev2 = prev3
#                 prev3 = prev4
#                 prev4 = clipped_recon_image
#                 if num == 10:
#                     break
#         else:
#             continue
#         break
        
    #             if cnt == 100:
    #                 break

    #         print("Test on Vimeo triplet dataset: model-{}".format(step))
    sumBpp /= cnt
    sumPsnr /= cnt
    sumMsssim /= cnt
    sumMsssimDB /= cnt
    logger.info("Dataset Average result---Dataset Num: {}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(cnt, sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
    print("Dataset Average result---Dataset Num: {}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(cnt, sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
    with open(text_name, 'a') as f_text:
            f_text.write("\n")
            f_text.write("Dataset Average result---Dataset Num: {}, Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(cnt, sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
            f_text.write("\n")
            f_text.write("\n")
#     sc.savemat('residuals_gt.mat', {'arr': residuals})

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
    model = VideoCoder(feature_channel, latent_channel)
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