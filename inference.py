import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests

from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='classical_sr')
    parser.add_argument('--scale', type=int, default=1,
                        help='scale factor: 1, 2, 3, 4, 8')
    parser.add_argument('--noise', type=int, default=15,
                        help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40,
                        help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=48,  # 128
                        help='patch size used in training SwinIR.')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/swinir_classical_sr_x3.pth')
    parser.add_argument('--folder_lq', type=str,
                        default='datasets/testing_lr_images',
                        help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None,
                        help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None,
                        help='Tile size, None for no tile during testing')
    parser.add_argument('--tile_overlap', type=int, default=32,
                        help='Overlapping of different tiles')
    parser.add_argument('--save_dir', type=str,
                        default='results/swinir_classical_sr_x3',
                        help='result image folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')

    model = define_model(args)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        imgname, img_lq = get_image_pair(args, path)
        # HCW-BGR to CHW-RGB
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1
                              else img_lq[:, :, [2, 1, 0]], (2, 0, 1))
        # CHW-RGB to NCHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat(
                [img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat(
                [img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            # CHW-RGB to HCW-BGR
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        cv2.imwrite(f'{save_dir}/{imgname}_pred.png', output)


def define_model(args):
    # 001 classical image sr
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3,
                    img_size=args.training_patch_size,
                    window_size=8, img_range=1.,
                    depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
                    num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
                    upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g]
                          if param_key_g in pretrained_model.keys()
                          else pretrained_model, strict=True)

    return model


def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = args.save_dir
        folder = args.folder_lq
        border = args.scale
        window_size = 8
    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    if args.task in ['classical_sr', 'lightweight_sr']:
        img_lq = cv2.imread(f'{args.folder_lq}/{imgname}{imgext}',
                            cv2.IMREAD_COLOR).astype(np.float32) / 255.

    return imgname, img_lq  # , img_gt


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, \
            "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)
                E[..., h_idx*sf:(h_idx+tile)*sf,
                  w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf,
                  w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()
