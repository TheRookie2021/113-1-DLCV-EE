# B: An improved model
# Implement an improved CNN-based model to perform segmentation.
#   ■ You may choose any model different from VGG16-FCN32s,
#   e.g., FCN16s, FCN8s, U-Net, SegNet, etc.)
#   ■ Using pre-trained models is allowed
#   ■ Transformer-based models are not allowed

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import os
import numpy as np
import random
from tqdm import tqdm
from PIL import Image 
# from torchvision import models
from utils import  train_save, find_best_model, parse_args
from utils_class import  LRASPP_MobileNet_V3, Semantic_Segmentation_Dataset
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

# from torchvision.io import read_image

#==== """"""""""""""""""""""""""""""""""""""""" Setting Config """"""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
  args=parse_args()
  result_log={}
  result_log.update({'config':args})
  # print(result_log)
  seed=args.seed
  random.seed(seed)
  torch.manual_seed(seed)
  generator = torch.Generator().manual_seed(seed)

  """ Data Folder """
  dataset_root=args.dataset_root  # '../hw1_data/p1_data/mini/train' 
  
  """ Model Folder """
  restart_ckpt=args.restart_finetune   
  
  """ Use GPU if available """
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  # print('Device used:', device)

  """ Config """
  experiment_code=args.experiment_code 
  epoch=args.epoch
  train_batch_size=args.train_batch_size
  test_batch_size=args.test_batch_size
  weight_decay=args.weight_decay
  lr=args.lr
  freeze=args.freeze_backbone

  """ Saving setting """
  save_interval=args.save_interval
  save_begin=args.save_begin
  save_dir_model=os.path.join(args.save_dir_model, experiment_code) # ! wrong path! should be finetune_on_BYOL
  save_dir_result=os.path.join(args.save_dir_result,experiment_code)
  

  """ Validation Transform setting """
  print(result_log)
  with open(f"{experiment_code}.txt", "a") as f:
    f.write(experiment_code + " finetune\n")
    f.write(str(result_log)+"\n")
    result_log.clear()
  
  # ==== """"""""""""""""""""""""""""""""""""""""" finetune phase """"""""""""""""""""""""""""""""""""""""""""""""""""***

  # step: LOAD the model
  model=LRASPP_MobileNet_V3(N_class=7).to(device)
  
  # model=lraspp_mobilenet_v3_large().to(device)
  # input=torch.ones(1,3,512,512,dtype=torch.float).to(device)
  # print(model)
  # out=model(input)
  # print(out.shape,out.dtype)
  
  #==== restart training  
  if restart_ckpt:
    print(f"loading {restart_ckpt} to continue fintuning")
    model.load_state_dict(torch.load(restart_ckpt, weights_only=True)['state_dict'])
  result_log.update({'model': model})

  #==== Freeze backbone  
  if freeze:
    for name, param in model.named_parameters():
      if name.startswith('backbone'):
        param.requires_grad = False
    

  # step: setup train/test dataset and dataloader 
  train_dataset=Semantic_Segmentation_Dataset(root=os.path.join(dataset_root,"train"))
  test_dataset=Semantic_Segmentation_Dataset(root=os.path.join(dataset_root,"validation"))
  train_dataloader=DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
  test_dataloader=DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
  print(len(train_dataloader))
  print(len(train_dataloader.dataset))
  print(len(test_dataloader))
  print(len(test_dataloader.dataset))

  # step: train it
  result_list=[]
  print("start training")
  train_save(model=model, device=device,
            epoch=epoch,
            lr=lr,
            weight_decay= weight_decay,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            save_interval=save_interval,
            save_dir=save_dir_model,
            save_begin=save_begin,
            result_log=result_list)
  result_log.update({'finetune_result': result_list})
  with open(f"{experiment_code}.txt", "a") as f:
    f.write(str(result_log))
    