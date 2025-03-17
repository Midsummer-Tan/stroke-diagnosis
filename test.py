import torch
import torch.nn as nn
from tqdm import tqdm
import torch.distributed as dist
import os

import utils
from static import IS_CUDA_AVAILABLE, IS_DISTRIBUTED

def test(val_loader, model):
    loss_fn = utils.CombineLoss()
    test_loss = utils.Averager()
    test_dice = utils.Averager()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='test', leave=False):
            model.eval()
            inp = batch['inp']
            label = batch['label']
            inp = inp.cuda()
            label = label.cuda()
            pred = model(inp)
            loss = loss_fn(pred, label)
            dice_fn = utils.DiceLoss()
            dice = 1 - dice_fn(pred, label).item()
            loss = loss.item()
            test_loss.add(loss)
            test_dice.add(dice)
    return test_loss.item(), test_dice.item()