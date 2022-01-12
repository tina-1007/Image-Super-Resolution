import os.path
from os.path import join
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils.utils_image import mkdirs, tensor2uint, imsave, calculate_psnr
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model

import wandb

'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path,
                        help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        mkdirs((path for key, path in opt['path'].items()
               if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(
        opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(
        opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E

    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(
        opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(
            logger_name, join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) /
                             dataset_opt['batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: \
                    {:,d}, iters: {:,d}'.format(len(train_set), train_size))

            if opt['dist']:
                train_sampler = DistributedSampler(
                    train_set, drop_last=True, seed=seed,
                    shuffle=dataset_opt['shuffle'],)

                train_loader = DataLoader(
                    train_set, batch_size=1, shuffle=False,
                    sampler=train_sampler,
                    num_workers=dataset_opt['num_workers']//opt['num_gpu'],
                    drop_last=True, pin_memory=True)
            else:
                train_loader = DataLoader(
                    train_set, batch_size=dataset_opt['batch_size'],
                    shuffle=dataset_opt['shuffle'],
                    num_workers=dataset_opt['num_workers'],
                    drop_last=True, pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    # if opt['rank'] == 0:
    #     logger.info(model.info_network())
    #     logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    # Train!
    wandb.init(project="HW4_Super_Resolution", entity="ytliu", config=opt)
    best_psnr = 0

    for epoch in range(100000):  # keep running
        for i, train_data in enumerate(train_loader):

            step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if step % opt['train']['freq_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)
                wandb.log({'iter': step, 'G_loss': v})

            # -------------------------------
            # 5) save model
            # -------------------------------
            if step % opt['train']['freq_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if step % opt['train']['freq_test'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = join(opt['path']['images'], img_name)
                    mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = tensor2uint(visuals['E'])
                    H_img = tensor2uint(visuals['H'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    save_img_path = join(
                        img_dir, '{:s}_{:d}.png'.format(img_name, step))
                    imsave(E_img, save_img_path)

                    # -----------------------
                    # calculate PSNR
                    # -----------------------
                    current_psnr = calculate_psnr(E_img, H_img, border=border)
                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(
                        idx, image_name_ext, step))

                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx

                # testing log
                msg = '<epoch:{:3d}, iter:{:8,d}, PSNR : {:<.2f}dB\n'.format(
                    epoch, step, avg_psnr)
                logger.info(msg)
                wandb.log({'PSNR': avg_psnr})

                if avg_psnr > best_psnr:
                    logger.info('Saving the model.')
                    model.save(step)

            # End Training
            if step > 200000:
                logger.info('******** End Training ********')
                return

if __name__ == '__main__':
    main()
