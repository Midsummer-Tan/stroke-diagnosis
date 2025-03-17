from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from torchvision import transforms
import torch

import utils

class NCCTDataset(Dataset):
    def __init__(self, args):
        self.path = args['path']
        self.images = os.listdir(os.path.join(self.path, "image"))
        self.masks = os.listdir(os.path.join(self.path, "mask"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_1 = transforms.ToTensor()(Image.open(os.path.join(self.path, "image", self.images[idx])))
        if idx==0 or self.images[idx-1][0:3]!=self.images[idx][0:3] or \
            int(self.images[idx-1][3:6])!=int(self.images[idx][3:6])-1:
            image_0 = image_1
        else:
            image_0 = transforms.ToTensor()(Image.open(os.path.join(self.path, "image", self.images[idx-1])))
        if idx == len(self.images)-1 or self.images[idx+1][0:3]!=self.images[idx][0:3] or \
            int(self.images[idx+1][3:6])!=int(self.images[idx][3:6])+1:
            image_2 = image_1
        else:
            image_2 = transforms.ToTensor()(Image.open(os.path.join(self.path, "image", self.images[idx+1])))
        image = torch.squeeze(torch.stack([image_0,image_1,image_2]))
        mask = torch.tensor(np.array(Image.open(os.path.join(self.path, "mask", self.masks[idx])), dtype=np.float64)).unsqueeze(0)
        return {
            'inp': image,
            'label': mask
        }