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
from utils_class import  VGG16_FCN32s, Semantic_Segmentation_Dataset
# from torchvision.io import read_image
from torchvision.models import resnet50

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
  model=VGG16_FCN32s().to(device)
  ## load parameters
  
  result_log.update({'model': model})

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
    