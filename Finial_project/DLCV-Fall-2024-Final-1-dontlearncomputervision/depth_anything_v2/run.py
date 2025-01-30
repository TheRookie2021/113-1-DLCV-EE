import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from PIL import Image
from datasets import load_dataset

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--split', type=str, choices=['train', 'validation', 'test'])
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    depth_anything = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_vitb.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # if os.path.isfile(args.img_path):
    #     if args.img_path.endswith('txt'):
    #         with open(args.img_path, 'r') as f:
    #             filenames = f.read().splitlines()
    #     else:
    #         filenames = [args.img_path]
    # else:
    #     filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # load dataset from the given path
    dataset = load_dataset("../data/dlcv_2024_final1", split=args.split)
    
    for i, data in enumerate(dataset):

        # if i == 1: break
        print(f"Generate depth map of the {i}th image. :)")

        raw_image = data['image']
        # print(type(raw_image)) # PIL.PngImagePlugin.PngImageFile

        # convert input type to numpy.ndarray
        raw_image = np.array(raw_image)

        depth = depth_anything.infer_image(raw_image, 518)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        # print("depth map shape:", depth.shape) # (720, 1355)
        """
        Each entry of depth is a number in [0, 255], which represents the depth information of the corresponding pixel.
        The larger the value, the closer it is to the camera.
        """

        # depth_transform = depth
        

        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        # print("depth map shape:", depth.shape) # (720, 1355, 3) 

        """
        Each entry of depth is an array of size 3, which represents the color value (BGR) of the corresponding pixel in the output picture.
        """
        
        cv2.imwrite(os.path.join(args.outdir, f'{args.split}_depth_image{i}.png'), depth)


        # print(depth_transform)
        # print("\n----------------------------------------------------- fuckUUUUUUUUUUUUUUUUUUUUUUUU -------------------------------------------------\n")

        # # Linearly transform each entry to the range [1, 4].
        # depth_transform = depth_transform // 64
        # depth_transform = depth_transform + 1

        # print(depth_transform)
