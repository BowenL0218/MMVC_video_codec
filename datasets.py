import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import PIL
import os

class VimeoDataset(Dataset):
    def __init__(self, video_dir, text_split, test=False, transform=None):
        """
        Dataset class for the Vimeo-90k dataset, available at http://toflow.csail.mit.edu/.
        Args:
            video_dir (string): Vimeo-90k sequences directory.
            text_split (string): Text file path in the Vimeo-90k folder, either `tri_trainlist.txt` or `tri_testlist.txt`.
            transform (callable, optional): Optional transform to be applied samples.
        """
        self.video_dir = video_dir
        self.test = test
        self.text_split = text_split
        # default transform as per RRIN, convert images to tensors, with values between 0 and 1
        if transform is None:# and test is False:
            self.transform = transforms.Compose([
#                 transforms.RandomCrop((256, 256)),
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
        self.cur_frame = []

        # open the given text file path that gives file names for train or test subsets
        with open(self.text_split, 'r') as f:
            filenames = f.readlines()
            f.close()
        full_filenames = []
        
        for i in filenames:
            full_filenames.append(os.path.join(self.video_dir, i.split('\n')[0]))

        for f in full_filenames:
            try:
                frames = [os.path.join(f, i) for i in os.listdir(f)]
            except:
                continue
            # make sure images are in order, i.e. im1.png, im2.png, im3.png
            frames = sorted(frames)
            # make sure there are only 3 images in the Vimeo-90k triplet's folder for it to be a valid dataset sample
            if len(frames) == 3:
                self.prev1_frame.append(frames[0])
                self.prev2_frame.append(frames[1])
                self.cur_frame.append(frames[2])
            if len(frames) == 7:
                for i in range(3):
                    self.prev1_frame.append(frames[i])
                    self.prev2_frame.append(frames[i+1])
                    self.prev3_frame.append(frames[i+2])
                    self.prev4_frame.append(frames[i+3])
                    self.cur_frame.append(frames[i+4])
#         print(frames)

    def __len__(self):
        return len(self.cur_frame)

    def __getitem__(self, idx):
        prev1 = PIL.Image.open(self.prev1_frame[idx]).convert("RGB")
        prev2 = PIL.Image.open(self.prev2_frame[idx]).convert("RGB")
        prev3 = PIL.Image.open(self.prev3_frame[idx]).convert("RGB")
        prev4 = PIL.Image.open(self.prev4_frame[idx]).convert("RGB")
        cur = PIL.Image.open(self.cur_frame[idx]).convert("RGB")

        if self.transform:
            prev1 = self.transform(prev1)
            prev2 = self.transform(prev2)
            prev3 = self.transform(prev3)
            prev4 = self.transform(prev4)
            cur = self.transform(cur)
        if self.test is False:
#             Concat = torch.cat([prev1, prev2, cur], axis = 0)
            Concat = torch.cat([prev1, prev2, prev3, prev4, cur], axis = 0)
#             print(Concat.shape)
            transform = transforms.Compose([
                transforms.RandomCrop((256, 256)),
#                 transforms.ToTensor(),
            ])
            Concat = transform(Concat)        
            prev1 = Concat[:3,:,:]
            prev2 = Concat[3:6,:,:]
            prev3 = Concat[6:9,:,:]
            prev4 = Concat[9:12,:,:]
            cur = Concat[12:,:,:]

#         return prev1, prev2, cur
        return prev1, prev2, prev3, prev4, cur
        