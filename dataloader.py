import torch 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import os 
import cv2 
import numpy as np

class PotteryDataset(Dataset):
    def __init__(self, datapath, transform=None):

        self.datapath = datapath
        self.img_list = [i[:-4] for i in os.listdir(datapath)]
        self.transform = transform

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.datapath, '%s.png'%self.img_list[idx])) 
        # img = np.transpose(img, (2,0,1))
        label = int(self.img_list[idx][-1])
        img_name = self.img_list[idx][:-2]
        if self.transform:
            img = self.transform(img)
        return img_name, img, label

    def __len__(self):
        return len(self.img_list)

