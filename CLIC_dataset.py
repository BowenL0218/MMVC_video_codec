import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import PIL
import os
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]



class CLICDataset(Dataset):
    def __init__(self, video_dir, test = False, transform=None):
        self.video_dir = video_dir
        self.test = test
        self.transform = transform
        if transform is None:# and test is False:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.prev1_frame = []
        self.prev2_frame = []
        self.prev3_frame = []
        self.prev4_frame = []
        self.prev5_frame = []
        self.cur_frame = []
        folder_list = sorted(os.listdir(self.video_dir))
        folder_list.sort(key=natural_keys)
        if test:
            for folder_name in folder_list[2:3]:
                folder_full_path = os.path.join(self.video_dir, folder_name)
                filenames_list = sorted(os.listdir(folder_full_path))
                filenames_list.sort(key=natural_keys)
                frames = [os.path.join(folder_full_path, i) for i in filenames_list]
                for i in range(len(frames) - 5):
                    self.prev1_frame.append(frames[i])
                    self.prev2_frame.append(frames[i+1])
                    self.prev3_frame.append(frames[i+2])
                    self.prev4_frame.append(frames[i+3])
                    self.prev5_frame.append(frames[i+4])
                    self.cur_frame.append(frames[i+5])
        else:
            for folder_name in folder_list[3:]:
                folder_full_path = os.path.join(self.video_dir, folder_name)
                filenames_list = sorted(os.listdir(folder_full_path))
                filenames_list.sort(key=natural_keys)
                frames = [os.path.join(folder_full_path, i) for i in filenames_list]
                for i in range(len(frames) - 5):
                    self.prev1_frame.append(frames[i])
                    self.prev2_frame.append(frames[i+1])
                    self.prev3_frame.append(frames[i+2])
                    self.prev4_frame.append(frames[i+3])
                    self.prev5_frame.append(frames[i+4])
                    self.cur_frame.append(frames[i+5])
            
        
    def __len__(self):
        return len(self.cur_frame)
    def __getitem__(self, idx):
        prev1 = PIL.Image.open(self.prev1_frame[idx]).convert("RGB")
        prev2 = PIL.Image.open(self.prev2_frame[idx]).convert("RGB")
        prev3 = PIL.Image.open(self.prev3_frame[idx]).convert("RGB")
        prev4 = PIL.Image.open(self.prev4_frame[idx]).convert("RGB")
        prev5 = PIL.Image.open(self.prev5_frame[idx]).convert("RGB")
        cur = PIL.Image.open(self.cur_frame[idx]).convert("RGB")

        if self.transform:
            prev1 = self.transform(prev1)
            prev2 = self.transform(prev2)
            prev3 = self.transform(prev3)
            prev4 = self.transform(prev4)
            prev5 = self.transform(prev5)
            cur = self.transform(cur)
        if self.test:
            prev1_stack = prev1[:, :1024, :1024]
            prev2_stack = prev2[:, :1024, :1024]
            prev3_stack = prev3[:, :1024, :1024]
            prev4_stack = prev4[:, :1024, :1024]
            prev5_stack = prev5[:, :1024, :1024]
            cur_stack = cur[:, :1024, :1024]
            
        else:
#             print(self.prev1_frame[idx])
#             print(self.prev5_frame[idx])

            _, H, W = prev1.shape
            x_cord_1 = H//2 - 256
            x_cord_2 = H//2 + 256
            y_cord_1 = W//2 - 256
            y_cord_2 = W//2 + 256
            prev1_stack = torch.stack((prev1[:, x_cord_1:x_cord_2, y_cord_1-100:y_cord_2-100], prev1[:, x_cord_1:x_cord_2, y_cord_1:y_cord_2], prev1[:, x_cord_1:x_cord_2, y_cord_1+100:y_cord_2+100]))
            prev2_stack = torch.stack((prev2[:, x_cord_1:x_cord_2, y_cord_1-100:y_cord_2-100], prev2[:, x_cord_1:x_cord_2, y_cord_1:y_cord_2], prev2[:, x_cord_1:x_cord_2, y_cord_1+100:y_cord_2+100]))
            prev3_stack = torch.stack((prev3[:, x_cord_1:x_cord_2, y_cord_1-100:y_cord_2-100], prev3[:, x_cord_1:x_cord_2, y_cord_1:y_cord_2], prev3[:, x_cord_1:x_cord_2, y_cord_1+100:y_cord_2+100]))
            prev4_stack = torch.stack((prev4[:, x_cord_1:x_cord_2, y_cord_1-100:y_cord_2-100], prev4[:, x_cord_1:x_cord_2, y_cord_1:y_cord_2], prev4[:, x_cord_1:x_cord_2, y_cord_1+100:y_cord_2+100]))
            prev5_stack = torch.stack((prev5[:, x_cord_1:x_cord_2, y_cord_1-100:y_cord_2-100], prev5[:, x_cord_1:x_cord_2, y_cord_1:y_cord_2], prev5[:, x_cord_1:x_cord_2, y_cord_1+100:y_cord_2+100]))
            cur_stack = torch.stack((cur[:, x_cord_1:x_cord_2, y_cord_1-100:y_cord_2-100], cur[:, x_cord_1:x_cord_2, y_cord_1:y_cord_2], cur[:, x_cord_1:x_cord_2, y_cord_1+100:y_cord_2+100]))
#             prev1_stack = torch.stack((prev1[:, x_cord_1:x_cord_2, y_cord_1-50:y_cord_2-50], prev1[:, x_cord_1:x_cord_2, y_cord_1:y_cord_2]))
#             prev2_stack = torch.stack((prev2[:, x_cord_1:x_cord_2, y_cord_1-50:y_cord_2-50], prev2[:, x_cord_1:x_cord_2, y_cord_1:y_cord_2]))
#             prev3_stack = torch.stack((prev3[:, x_cord_1:x_cord_2, y_cord_1-50:y_cord_2-50], prev3[:, x_cord_1:x_cord_2, y_cord_1:y_cord_2]))
#             prev4_stack = torch.stack((prev4[:, x_cord_1:x_cord_2, y_cord_1-50:y_cord_2-50], prev4[:, x_cord_1:x_cord_2, y_cord_1:y_cord_2]))
#             prev5_stack = torch.stack((prev5[:, x_cord_1:x_cord_2, y_cord_1-50:y_cord_2-50], prev5[:, x_cord_1:x_cord_2, y_cord_1:y_cord_2]))
#             cur_stack = torch.stack((cur[:, x_cord_1:x_cord_2, y_cord_1-50:y_cord_2-50], cur[:, x_cord_1:x_cord_2, y_cord_1:y_cord_2]))
#         print(prev1_stack.shape)
#         prev1_list = torch.stack((prev1[:, ]))
#         if self.test is False:
# #             Concat = torch.cat([prev1, prev2, cur], axis = 0)
#             Concat = torch.cat([prev1, prev2, prev3, prev4, prev5, cur], axis = 0)
# #             print(Concat.shape)
#             transform = transforms.Compose([
#                 transforms.RandomCrop((256, 256)),
# #                 transforms.ToTensor(),
#             ])
#             Concat = transform(Concat)        
#             prev1 = Concat[:3,:,:]
#             prev2 = Concat[3:6,:,:]
#             prev3 = Concat[6:9,:,:]
#             prev4 = Concat[9:12,:,:]
#             cur = Concat[12:,:,:]

#         return prev1, prev2, cur
        return prev1_stack, prev2_stack, prev3_stack, prev4_stack, prev5_stack, cur_stack


# clic_dir = '/nfs/turbo/coe-hunseok/mrakeshc/clic2022'
# train_dataset = CLICDataset(clic_dir)
# train_loader = DataLoader(dataset=train_dataset,
#                               batch_size=2,
#                               shuffle=True,
#                               pin_memory=True,
#                               num_workers=1)
# for batch_idx, (prev1, prev2, prev3, prev4, prev5, cur) in enumerate(train_loader):
#     print(prev1.shape)
# it = iter(train_loader)
# print(next(it).shape)