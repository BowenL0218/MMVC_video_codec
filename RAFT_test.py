import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from RAFT_master.warping import warp, torch_warp

from RAFT_master.core.raft import RAFT
# from utils import flow_viz
from RAFT_master.core.utils.utils import InputPadder
import matplotlib.pyplot as plt



DEVICE = 'cuda'

# def load_image(imfile):
    
#     img = np.array(Image.open(imfile)).astype(np.uint8)
#     img = torch.from_numpy(img).permute(2, 0, 1).float()
#     # img = Image.open(imfile).convert("RGB")
#     # transform = transforms.Compose([
#     #             transforms.ToTensor(),
#     #         ])
#     # img = torch.from_numpy(img).float()
#     # img = transform(img)
#     # img = img*255
#     # print(torch.max(img))
#     # print(torch.min(img))
#     return img[None].to(DEVICE)


# def viz(img, flo, path):
#     img = img[0].permute(1,2,0).cpu().numpy()
#     flo = flo[0].permute(1,2,0).cpu().numpy()
#     # print(flo.shape)
    
#     # map flow to rgb image
#     flo = flow_viz.flow_to_image(flo)
#     img_flo = np.concatenate([img, flo], axis=0)

#     # plt.imshow(img_flo / 255.0)
#     # plt.show()
#     # print(flo.shape)
#     plt.imsave(path, flo / 255.0)
# #     cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
# #     cv2.waitKey()


def demo(image1, image2):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/home/mrakeshc/RAFT_video_compression_rev/RAFT_master/models/raft-things.pth', help="restore checkpoint")
    # parser.add_argument('--path', default='/home/mrakeshc/Video_compression/RAFT-master/demo-frames/', help="dataset for evaluation")
    # parser.add_argument('--path', default='/z/bowenliu/vimeo_triplet/sequences/00001/0001/', help="dataset for evaluation")
    parser.add_argument('--small', default = False, action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    image1 = image1 * 255
    image2 = image2 * 255
    model = torch.nn.DataParallel(RAFT(args))
#     model = RAFT(args)

    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        # images = glob.glob(os.path.join(args.path, '*.png')) + \
        #          glob.glob(os.path.join(args.path, '*.jpg'))
        # images = sorted(images)
        # print(images)
        # for imfile1, imfile2 in zip(images[:-1], images[1:]):
        #     image1 = load_image(imfile1)
        #     # print(image1.shape)
        #     image2 = load_image(imfile2)

            # print(imfile2)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        # viz(image1, flow_up, imfile2.split('/')[-1])
        out = warp(image2, -flow_up)
        # out = torch_warp(image2, flow_up)
        # out = torch.squeeze(out)
        out = torch.clip(out/255, 0, 1)
        # out = out.permute(1,2,0)
        # out = out.cpu()
        # out = np.float32(out)
        # out = np.clip(out/255, 0, 1)
        # print(np.max(out))
        # plt.imsave('test.png', out)
    return out


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', default='models/raft-things.pth', help="restore checkpoint")
#     parser.add_argument('--path', default='/home/mrakeshc/Video_compression/RAFT-master/demo-frames/', help="dataset for evaluation")
#     # parser.add_argument('--path', default='/z/bowenliu/vimeo_triplet/sequences/00001/0001/', help="dataset for evaluation")
#     parser.add_argument('--small', default = False, action='store_true', help='use small model')
#     parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
#     parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
#     args = parser.parse_args()

#     demo(args)
