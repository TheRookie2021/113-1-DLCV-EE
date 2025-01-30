import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import os 
import numpy as np
from PIL import Image
from utils_class import  LRASPP_MobileNet_V3

# import csv

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='HW1 deep learning network.')
    parser.add_argument(
        '--load_dataset_dir', default='', type=str, help='')
    parser.add_argument(
        '--load_model', default='', type=str, help='')
    
    parser.add_argument(
        '--save_output', default='./p1/record', type=str, help='')
    
    # --load_dataset_dir $1 --save_output $2 --load_model
    return parser.parse_args()

def torch_to_PIL(img):
    img = img.detach().cpu().numpy()[0]
    mask = np.zeros((512, 512, 3), dtype=np.uint8)
    # print(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j]==0: mask[i][j] = [0, 255, 255]
            if img[i][j]==1: mask[i][j] = [255, 255, 0]
            if img[i][j]==2: mask[i][j] = [255, 0, 255]
            if img[i][j]==3: mask[i][j] = [0, 255, 0]
            if img[i][j]==4: mask[i][j] = [0, 0, 255]
            if img[i][j]==5: mask[i][j] = [255, 255, 255]
            if img[i][j]==6: mask[i][j] = [0, 0, 0]
    RGB_mask = Image.fromarray(mask.astype('uint8'), 'RGB')
    return RGB_mask

def inference(model, device, img):
    img=img.to(device)
    img=img[None,:,:,:]
    output=model(img)
    pred=output.argmax(dim=1)
    return pred

args=parse_args()
load_dataset_dir=args.load_dataset_dir
load_model=args.load_model
save_output=args.save_output
transform =  transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(   mean=[0.485, 0.456, 0.406],
                                        std =[0.229,0.224,0.225] )  ]) 
        
files= [f for f in os.listdir(load_dataset_dir) if f.endswith(".jpg") and "sat" in f]
device="cuda"

# step: load model
model=LRASPP_MobileNet_V3().to(device)
model.load_state_dict(torch.load(load_model, weights_only=True)['state_dict'])
model.eval()
# step: read image
for f in files:
    img=Image.open(os.path.join(load_dataset_dir,f))
    img=transform(img)
    print(f"read: {f}")
    f=f.split("_")[0]
    # step: inference
    pred=inference(model,device,img)
    torch_to_PIL(pred).save(f"{os.path.join(save_output,f)}_mask.png")
    print(f"saved at: {os.path.join(save_output,f)}_mask.png")


