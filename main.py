# modified from: https://github.com/jaewon-lee-b/lte
### CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 main.py
### Using specified GPU: 
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.run --nproc_per_node=1 main.py
### make progress still run after close vscode
# CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.run --nproc_per_node=2 main.py &
# When using linux, change to dist.init_process_group(backend='nccl')
import argparse
import os
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import torch.distributed as dist
import time

from static import IS_CUDA_AVAILABLE, IS_DISTRIBUTED
import utils
import test
import train
from model import model_mymodel


if __name__ == '__main__':
    config_path = './setting.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=config_path)
    parser.add_argument('--main_gpu', default = 0, type = int)
    # main-gpu use to save and load model
    args = parser.parse_args()

    distributed_on_other_gpu = False
    if IS_DISTRIBUTED:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group(backend='nccl')
        distributed_on_other_gpu = IS_DISTRIBUTED and dist.get_rank() != 0
        # windows: dist.init_process_group(backend="gloo", init_method="env://") 
        # linux: dist.init_process_group(backend='nccl')
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    epoch_max = config['epoch_max']
    timer = utils.Timer()
    run_time = time.strftime('%Y%m%d%H%M%S')
    log_save_path = r"./save/log"
    logger = utils.logger_config(log_path=os.path.join(log_save_path, f'test_log_{run_time}.log'), logging_name="myunet")
    source = {} # train_dataset, test_dataset, train_sampler, test_sampler, train_loader, test_loader
    for i in ['train', 'test']:
        args = config[f'{i}_dataset']
        per_dataset = utils.make_dataset(args)
        source[f'{i}_dataset'] = per_dataset
        sampler = utils.make_sampler(per_dataset, type=i)
        source[f'{i}_sampler'] = sampler
        loader = utils.make_loader(per_dataset, sampler, args)
        source[f'{i}_loader'] = loader

    model, optimizer, lr_scheduler, dice_max, epoch_start = utils.prepare_training()

    for epoch in range(epoch_start, epoch_max + 1):
        t_start = timer.t()
        if not distributed_on_other_gpu:
            log_info = 'epoch {}/{}'.format(epoch, epoch_max)
            logger.info(log_info)
        if IS_DISTRIBUTED:
            source['train_sampler'].set_epoch(epoch=epoch)
        train_loss, train_dice = train.train(source['train_loader'], model, optimizer)
        if not distributed_on_other_gpu: # DDP模式其他进程运行时不log
            utils.log_metrics(logger, train_loss, train_dice, 'train')
        test_loss, test_dice = test.test(source['test_loader'], model)
        if not distributed_on_other_gpu:
            utils.log_metrics(logger, test_loss, test_dice, 'test')
        if test_dice > dice_max:
            if IS_DISTRIBUTED and dist.get_rank() == 0:
                utils.save_best_model(model.module, 'dice')
            elif not IS_DISTRIBUTED:
                utils.save_best_model(model, 'dice')
            dice_max = test_dice
            epoch_dice_best = epoch
            if not distributed_on_other_gpu:
                log_info = f'Best dice: epoch [{epoch_dice_best}]'
                logger.info(log_info)
        if IS_DISTRIBUTED and dist.get_rank() == 0:
            utils.save_model(model.module, optimizer, lr_scheduler, epoch, dice_max)
        elif not IS_DISTRIBUTED:
            utils.save_model(model, optimizer, lr_scheduler, epoch, dice_max)
        t_stop = timer.t()
        progress = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_transfer(t_stop - t_start)
        t_elapsed, t_all = utils.time_transfer(t_stop), utils.time_transfer(t_stop / progress)
        if not distributed_on_other_gpu:
            log_info = '{} {}/{}'.format(t_epoch, t_elapsed, t_all)
            logger.info(log_info)
        if IS_DISTRIBUTED:
            dist.barrier()
    if IS_DISTRIBUTED:
        dist.destroy_process_group()
