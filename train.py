# modified from: https://github.com/jaewon-lee-b/lte
import argparse
import os
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import torch.distributed as dist
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR
import time

import utils
import test
from model import model_mymodel
from static import IS_CUDA_AVAILABLE, IS_DISTRIBUTED

def train(train_loader, model, optimizer):
    loss_fn = utils.CombineLoss()
    # loss_fn = utils.FocalLoss()
    if IS_DISTRIBUTED:
        loss_fn.to(dist.get_rank())
    elif IS_CUDA_AVAILABLE:
        loss_fn.cuda()
    train_loss = utils.Averager()
    train_dice = utils.Averager()

    for batch in tqdm(train_loader, desc='train', leave=False):
        model.train()
        inp = batch['inp']
        label = batch['label']
        if IS_DISTRIBUTED:
            inp = inp.to(dist.get_rank())
            label = label.to(dist.get_rank())
        elif IS_CUDA_AVAILABLE:
            inp = inp.cuda()
            label = label.cuda()
        pred = model(inp)
        loss = loss_fn(pred, label)
        dice_fn = utils.DiceLoss()
        dice = 1 - dice_fn(pred, label).item()
        train_loss.add(loss.item())
        train_dice.add(dice)

        optimizer.zero_grad()
        # loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

    return train_loss.item(), train_dice.item()