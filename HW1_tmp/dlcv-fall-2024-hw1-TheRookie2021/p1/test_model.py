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
from utils import BYOL_train_save, train_save, find_best_model, parse_args, test
from utils_class import Mini_ImageNet_unlabelled, Office_Dataset, FinetuneResNet
# from torchvision.io import read_image
from torchvision.models import resnet50

# """"""""""""""""""""""""""""""""""""""""" Setting Config """"""""""""""""""""""""""""""""""""""""""""""""""""
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
  pretrain_dataset_root=args.pretrain_dataset_root  # '../hw1_data/p1_data/mini/train' 
  fintune_dataset_root=args.fintune_dataset_root   # '../hw1_data/p1_data/office'

  """ Model Folder """
  pretrain_path=args.pretrain_path   
  model_to_load=args.inference_model
  
  """ Use GPU if available """
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  # print('Device used:', device)

  """ Config """
  experiment_code=args.experiment_code # '01-03-v2-adam'
  epoch_pretrain=args.epoch_pretrain # 50
  epoch_finetune=args.epoch_finetune # 200
  train_batch_size=args.train_batch_size
  test_batch_size=args.test_batch_size
  weight_decay=args.weight_decay
  lr=args.lr

  """ Saving setting """
  save_interval=args.save_interval
  save_begin=args.save_begin
  save_dir_pretrain_model=os.path.join(args.save_dir_pretrain_model, '01')
  save_dir_finetune_model=os.path.join(args.save_dir_finetune_model, experiment_code) # ! wrong path! should be finetune_on_BYOL
  save_dir_result=os.path.join(args.save_dir_pretrain_model,experiment_code)
  
  """ Validation Transform setting """
  VAL_TRANSFORM= transforms.Compose([
            transforms.Resize(128),                            
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                  std =[0.229,0.224,0.225] )
        ]) # by hw1 constrain
 
  # step: LOAD the model
  Finetune_model=FinetuneResNet().to(device)
  Finetune_model.load_state_dict(torch.load(model_to_load, weights_only=True)['state_dict'])

  # print(Finetune_model)
  

  print(f"using {model_to_load}")
  # step: setup train/test dataset and dataloader 
  test_dataset=Office_Dataset(annotations_file=os.path.join(fintune_dataset_root,'val.csv'), transform=VAL_TRANSFORM, img_dir=os.path.join(fintune_dataset_root,'val') )
  test_dataloader=DataLoader(test_dataset,batch_size=test_batch_size, shuffle=True)
  print(len(test_dataloader))
  print(len(test_dataloader.dataset))

  # step: test it
  loss,acc= test(Finetune_model,device, test_dataloader, [])
  
  result_log.update({"model name:":args.restart_finetune })
  result_log.update({"loss:":loss})
  result_log.update({"acc:":acc})

  with open(f"test_model.txt", "a") as f:
    f.write(str(result_log))
    