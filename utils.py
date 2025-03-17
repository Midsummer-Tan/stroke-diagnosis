# modified from: https://github.com/jaewon-lee-b/lte

import os
import time
import math
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler,SequentialSampler
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR
from torch.utils.data.distributed import DistributedSampler
import seaborn as sns
from matplotlib import pyplot as plt

import dataset
from model import model_mymodel
from static import IS_CUDA_AVAILABLE, IS_DISTRIBUTED

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        num = targets.size(0)
        flat_inputs = inputs.view(num, -1)
        flat_targets = targets.view(num, -1)
        intersection = torch.sum(flat_inputs * flat_targets, dim=1)
        union = torch.sum(flat_inputs, dim=1) + torch.sum(flat_targets, dim=1)
        dice_scores = 2 * intersection / (union + 1e-8)
        return 1 - dice_scores.mean()

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        logp = self.ce(inputs, targets)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class CombineLoss(nn.Module):

    def __init__(self):
        super(CombineLoss, self).__init__()
        self.dl = DiceLoss()
        self.fl = FocalLoss()

    def forward(self, inputs, targets, alpha = 1, beta = 1):
        loss = alpha*self.dl(inputs, targets) + beta*self.fl(inputs, targets)
        return loss

def make_dataset(args):
    args = args['dataset']['args']
    ct_dataset = dataset.NCCTDataset(args)
    return ct_dataset

def make_sampler(dataset, type):
    if IS_DISTRIBUTED and type == 'train':
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = SequentialSampler(dataset) #RandomSampler or SequentialSampler,origin is random
    return sampler

def make_loader(dataset, sampler, args):
    batch_size = args['batch_size']
    loader = DataLoader(dataset, batch_size, sampler = sampler,
        shuffle=False, num_workers=1, pin_memory=True)
    return loader

def prepare_training():
    model = model_mymodel.MyModel()
    if IS_DISTRIBUTED: # IS_DISTRIBUTED = True indicated IS_CUDA_AVAILABLE=True
        model = model.to(dist.get_rank())
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[dist.get_rank()], find_unused_parameters=False) # Set to True to find unused params
    elif IS_CUDA_AVAILABLE:
        model = model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # lr_scheduler = ExponentialLR(optimizer, gamma=0.95)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=-1)
    epoch_start = 1
    dice_max = 0

    model_save_path = r"./save/model"
    latest_model_path = os.path.join(model_save_path, "epoch-latest.pth")
    if os.path.isfile(latest_model_path):
        checkpoint = torch.load(latest_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        dice_max = checkpoint['dice_max']
        epoch_start = checkpoint['epoch'] + 1
        print(epoch_start)

    return model, optimizer, lr_scheduler, dice_max, epoch_start

class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v

class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

def time_transfer(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))

def save_model(model, optimizer, lr_scheduler, epoch, dice_max):
    save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch,
                'dice_max':dice_max}
    torch.save(save_dict, os.path.join("./save/model", 'epoch-latest.pth'))

def save_best_model(model, type):
    torch.save(model.state_dict(), os.path.join("./save/model", f'best-{type}-model.pth'))

def save_model_structure(model):
    print(model)
    with open("./save/model/model.txt", "w") as file:
        file.write(str(model))



def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def log_metrics(logger, loss, dice, type):
    log_info = '{}: loss={:.4f}'.format(type, loss)
    logger.info(log_info)
    log_info = '{}: dice={:.4f}'.format(type, dice)
    logger.info(log_info)

def draw_heatmap(inp):
    save_path = "./save/heatmap"
    file_list = os.listdir(save_path)
    name_pre = 1
    name_post = 0
    if file_list:
        file_list = sorted(file_list,key=lambda x: os.path.getmtime(os.path.join(save_path, x)))
        name_pre = int(file_list[-1].split("_")[0])
        name_post = int(file_list[-1].split("_")[1][0:-4])
    ax = sns.heatmap(inp, cmap="YlGnBu", xticklabels=[], yticklabels=[])
    if name_post == 9:
        name_post = 1
        name_pre += 1
    else:
        name_post+=1
    new_name = str(name_pre)+"_"+str(name_post)
    print(new_name)
    plt.show()
    #plt.savefig(os.path.join(save_path, new_name))
    plt.close()

def show_plt(tensor_data):
    # e is a tensor
    numpy_data = tensor_data.detach().cpu().numpy()
    plt.imshow(numpy_data,cmap='gray')
    plt.axis('off')
    plt.show()

def get_mask_graph(path):
    img = Image.open(os.path.join(path))
    h, w = img.shape[0:2]
    h1,w1 = int(h/2), int(w/2)
    size = 10
    img[h1-int(size/2):h1+int(size/2), w1-int(size/2):w1+int(size/2)] = 0
    show_plt(transforms.ToTensor()(img))