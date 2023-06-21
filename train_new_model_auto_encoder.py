import os
import argparse
from model_auto_encoder import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import random
from CLIC_auto_dataset import CLICDataset
from tensorboardX import SummaryWriter
from Meter import AverageMeter
import numpy as np
from PIL import Image



torch.backends.cudnn.enabled = True
# gpu_num = 4
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 2e-5#  * gpu_num

train_lambda = 2048
print_freq = 100
cal_step = 40
warmup_step = 0#  // gpu_num
batch_size = 16
tot_epoch = 1000000
tot_step = 2500000
decay_interval = 2200000
lr_decay = 0.1
image_size = 256
logger = logging.getLogger("VideoCompression")
global_step = 0
save_model_freq = 50000
test_step = 10000
feature_channel = 96
latent_channel = 192
alpha = 0.75
parser = argparse.ArgumentParser(description='A video coder toy model')

parser.add_argument('-n', '--name', default='deep_encoder_auto_encoder_latent_512_512_3_64_64_96_no_res_no_sigmoid', help='experiment name')
parser.add_argument('-p', '--pretrain', default='/nfs/turbo/coe-hunseok/mrakeshc/vimeo_triplet_video_compression/deep_encoder_auto_encoder_latent_512_512_3_64_64_96_no_res_no_sigmoid/iter_184015.pth.tar', help='load pretrain model')
parser.add_argument('--test', action='store_true')
parser.add_argument('--config', dest='config', default = '/home/mrakeshc/RAFT_video_compression/configs/pixelcnn_1conv.json', required=False, help='hyperparameter in json format')
parser.add_argument('--seed', default=234, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--dataset', dest='dataset', default = '/nfs/turbo/coe-hunseok/mrakeshc/clic2022/', required=False, help='the path of training dataset')


def parse_config(config):
    config = json.load(open(args.config))
    global tot_epoch, tot_step, base_lr, cur_lr, lr_decay, decay_interval, train_lambda, batch_size, print_freq, \
        out_channel_M, out_channel_N, save_model_freq, test_step
    if 'tot_epoch' in config:
        tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        tot_step = config['tot_step']
    if 'batch_size' in config:
        batch_size = config['batch_size']
    if "print_freq" in config:
        print_freq = config['print_freq']
    if "test_step" in config:
        test_step = config['test_step']
    if "save_model_freq" in config:
        save_model_freq = config['save_model_freq']
    if 'lr' in config:
        if 'base' in config['lr']:
            base_lr = config['lr']['base']
            cur_lr = base_lr
        if 'decay' in config['lr']:
            lr_decay = config['lr']['decay']
        if 'decay_interval' in config['lr']:
            decay_interval = config['lr']['decay_interval']


def adjust_learning_rate(optimizer, global_step, epoch):
#     print("This is the epoch number", epoch)
    global cur_lr
    global base_lr
    global warmup_step
#     print("This is the current lr", base_lr)
    if global_step < warmup_step:
        lr = base_lr * global_step / warmup_step
    elif global_step < decay_interval:#  // gpu_num:
        lr = base_lr
    elif epoch % 40 == 0:
        lr = base_lr * lr_decay
#         print("This is the learning rate after decay:", lr)
    else:
        # lr = base_lr * (lr_decay ** (global_step // decay_interval))
        lr = base_lr
#         print("This is the learning rate we are using:", lr)
    cur_lr = lr
    base_lr = lr
#     print("This is the base_lr learning rate we are using:", base_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test(step):
    with torch.no_grad():
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        for batch_idx, prev in enumerate(test_loader):
            batch_size, channels, H, W = prev.shape
#             recon_cur, probs_features, prob_z = net(prev1.to('cuda'), prev2.to('cuda'), cur.to('cuda'))
            recon_cur = net(prev.to('cuda'))
#             clipped_recon_image = recon_cur
            clipped_recon_image = recon_cur.clamp(0., 1.)
            mse_loss = torch.mean((clipped_recon_image - prev.to('cuda')).pow(2))
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumPsnr += psnr
            cpu_recon_image = clipped_recon_image.cpu().detach()
           
            np_recon_image = cpu_recon_image.permute(0,2,3,1).squeeze().numpy()
            image = Image.fromarray((np_recon_image*255).astype(np.uint8))
            image.save('/nfs/turbo/coe-hunseok/mrakeshc/recon/recon_{}.png'.format(cnt+1))
            orig_image = prev.permute(0,2,3,1).squeeze().numpy()
            orig_image = Image.fromarray((orig_image*255).astype(np.uint8))
            orig_image.save('/nfs/turbo/coe-hunseok/mrakeshc/recon/orig_{}.png'.format(cnt+1))
            msssim = ms_ssim(cpu_recon_image, prev, data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            cnt += 1
            logger.info("Num: {}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(cnt, psnr, msssim, msssimDB))
            if cnt == 2:
                break

        logger.info("Test on Vimeo triplet dataset: model-{}".format(step))
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        logger.info("Dataset Average result---Dataset Num: {}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(cnt, sumPsnr, sumMsssim, sumMsssimDB))


def train(epoch, global_step):
    logger.info("Epoch {} begin".format(epoch))
    net.train()
    global optimizer
    elapsed, losses, psnrs, mse_losses = [AverageMeter(print_freq) for _ in range(4)]
    for batch_idx, prev in enumerate(train_loader):
        batch_size, seq, channels, H, W = prev.shape
        start_time = time.time()
        global_step += 1
        recon_cur = net(prev.reshape(-1, channels, H, W).to('cuda'))
        clipped_recon_image = recon_cur.clamp(0., 1.)
#         print(clipped_recon_image.shape)
#         print(prev.shape)
        mse_loss = torch.mean((clipped_recon_image - prev.reshape(-1, channels, H, W).to('cuda')).pow(2))
#         print(mse_loss)
        
        distortion = mse_loss
        rd_loss = distortion
        optimizer.zero_grad()
        rd_loss.backward()

        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)
        clip_gradient(optimizer, 5)
        optimizer.step()

        # model_time += (time.time()-start_time)
        if (global_step % cal_step) == 0:
            # t0 = time.time()
            if mse_loss.item() > 0:
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
                psnrs.update(psnr.item())
            else:
                psnrs.update(100)
            # t1 = time.time()
            elapsed.update(time.time() - start_time)
            losses.update(rd_loss.item())
            mse_losses.update(mse_loss.item())

        if (global_step % print_freq) == 0:
            process = global_step / tot_step * 100.0
            log = (' | '.join([
                f'Step [{global_step}/{tot_step}={process:.2f}%]',
                f'Epoch {epoch}', f'Time {elapsed.val:.3f} ({elapsed.avg:.3f})', 
                f'Lr {cur_lr}', f'Total Loss {losses.val:.3f} ({losses.avg:.3f})', 
                f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})', 
                f'MSE {mse_losses.val:.5f} ({mse_losses.avg:.5f})',
                ]))
            logger.info(log)
        mse_loss_list.append(mse_loss.item())
        rd_loss_list.append(rd_loss.item())
        if mse_loss.item() > 0:
            signal_to_noise = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
            signal_to_noise_list.append(signal_to_noise.item())
#         print(outfile)
        if (global_step % print_freq) == 0:
            np.savez(outfile, np.array(mse_loss_list), np.array(rd_loss_list), np.array(signal_to_noise_list))
        if (global_step % save_model_freq) == 0:
            save_model(model, global_step, save_path)

    return global_step

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(seed=args.seed)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    tb_logger = None
    save_path = os.path.join('/nfs/turbo/coe-hunseok/mrakeshc/vimeo_triplet_video_compression/deep_encoder_auto_encoder_latent_512_512_3_64_64_96_no_res_no_sigmoid ', args.name)
    if args.name != '':
        os.makedirs(save_path, exist_ok=True)
        filehandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("video compression training")
    logger.info("config : ")
    logger.info(open(args.config).read())
    parse_config(args.config)
    model = VideoCoder(batch_size, feature_channel, latent_channel)
    if args.pretrain != '':
        logger.info("loading model:{}".format(args.pretrain))
        global_step = load_model(model, args.pretrain)
        
    net = model.cuda()
    net = torch.nn.DataParallel(net, list(range(gpu_num)))
    parameters = net.parameters()
    global test_loader
    test_dataset = CLICDataset(args.dataset, test = True)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True)

    
    optimizer = optim.Adam(parameters, lr=base_lr)
    global train_loader
    mse_loss_list = []
    rd_loss_list = []
    bpp_list = []
    bpp_feature_list = []
    bpp_z_list = []
    signal_to_noise_list = []

    outfile = os.path.join(save_path, 'losses.npz')
    train_dataset = CLICDataset(args.dataset)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=1)
    steps_epoch = global_step // (len(train_dataset) // (batch_size))
    save_model(model, global_step, save_path)
    for epoch in range(steps_epoch, tot_epoch):
        test(global_step)
        adjust_learning_rate(optimizer, global_step, epoch)
        if global_step > tot_step:
            save_model(model, global_step, save_path)
            break
        global_step = train(epoch, global_step)
        save_model(model, global_step, save_path)
