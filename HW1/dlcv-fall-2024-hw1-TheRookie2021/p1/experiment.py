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
from utils import BYOL_train_save, train_save, find_best_model, parse_args
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
 
  print(result_log)
  with open(f"{experiment_code}.txt", "a") as f:
    f.write(experiment_code + " finetune\n")
    f.write(str(result_log)+"\n")
    result_log.clear()
  
  # ***""""""""""""""""""""""""""""""""""""""""" pretrain phase (pretrain from scretch) """"""""""""""""""""""""""""""""""""""""""""""""""""***
  if 'C' in experiment_code:
    # setup model
    save_dir_pretrain_model=os.path.join(args.save_dir_pretrain_model,experiment_code)
    model = resnet50(weights=None).to(device)
    result_log.update({'pretrain_model': model})

    # setup train/test dataset and dataloader 
    pretrain_dataset=Mini_ImageNet_unlabelled(pretrain_dataset_root)
    train_dataloader=DataLoader(pretrain_dataset,batch_size=train_batch_size, shuffle=True)
    print(len(train_dataloader))
    print(len(train_dataloader.dataset))

    # train it 
    result_list=[]
    BYOL_train_save(model=model, device=device,
                    epoch=epoch_pretrain,
                    lr=lr, 
                    train_dataloader=train_dataloader,
                    save_interval=save_interval, save_dir=save_dir_pretrain_model, result_log=result_list)
    # save result log
    result_log.update({'pretrain_result': result_list})
    with open(f"{experiment_code}.txt", "a") as f:
      f.write(str(result_log)+"\n")
      result_log.clear()

  # ***""""""""""""""""""""""""""""""""""""""""" finetune phase """"""""""""""""""""""""""""""""""""""""""""""""""""***
  # step LOAD the model
  Finetune_model=FinetuneResNet().to(device) #? experiment A: do not pretrain a backbone model
  
  """use pretrained model as backbone"""
  if not 'A' in experiment_code:  
    ## fine target .pth to load
    if args.restart_finetune == '':
      # note: get targeted backbone model, check if pretrain_path is assigned as dir or as .pth file or not assigned
      if(pretrain_path == ''):  
        pretrain_model=find_best_model(save_dir_pretrain_model)
      else: 
        if(os.path.isdir(pretrain_path)):
          print(f"\nfinding best model in {pretrain_path}")
          pretrain_model=find_best_model(pretrain_path)
        else:
          pretrain_model=pretrain_path
      print(f"using {pretrain_model} as pretrained backbone") 
      result_log.update({'fintune_on': pretrain_model})

      # update param: Construct a new state dict in which the layers we want to import from the checkpoint is update with the parameters from the checkpoint
      ckpt=torch.load(pretrain_model)
      print(ckpt)
      states_to_load = {}

      # note: do not update the fc layers of resnet
      if 'B' in experiment_code or 'D' in experiment_code: 
        # note: TAs' model
        for name, param in ckpt.items():
            # print(name)
            if not name.startswith('fc'):
                states_to_load[name] = param
      
      elif 'C' in experiment_code or 'E' in experiment_code: 
        # note: my SSL pretrain model
        for name, param in ckpt['state_dict'].items():
            # print(name)
            if not name.startswith('fc'):
                states_to_load[name] = param
      model_state = Finetune_model.backbone.state_dict()
      model_state.update(states_to_load)
      Finetune_model.backbone.load_state_dict(model_state)
    else:
      Finetune_model.load_state_dict(torch.load(args.restart_finetune, weights_only=True)['state_dict'])
    
    result_log.update({'finetune_model': Finetune_model})

  
  
  #==== Freeze backbone  
  if 'D' in experiment_code or 'E' in experiment_code: 
    for name, param in Finetune_model.named_parameters():
      if name.startswith('backbone'):
        param.requires_grad = False

  # step setup train/test dataset and dataloader 
  train_dataset=Office_Dataset(annotations_file=os.path.join(fintune_dataset_root,'train.csv'),img_dir=os.path.join(fintune_dataset_root,'train'))
  test_dataset=Office_Dataset(annotations_file=os.path.join(fintune_dataset_root,'val.csv'), transform=VAL_TRANSFORM, img_dir=os.path.join(fintune_dataset_root,'val') )
  train_dataloader=DataLoader(train_dataset,batch_size=train_batch_size, shuffle=True)
  test_dataloader=DataLoader(test_dataset,batch_size=test_batch_size, shuffle=True)
  print(len(train_dataloader))
  print(len(train_dataloader.dataset))
  print(len(test_dataloader))
  print(len(test_dataloader.dataset))

  # step train it
  result_list=[]
  train_save(model=Finetune_model, device=device,
            epoch=epoch_finetune,
            lr=lr,
            weight_decay= weight_decay,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            save_interval=save_interval,
            save_dir=save_dir_finetune_model,
            save_begin=save_begin,
            result_log=result_list)
  result_log.update({'finetune_result': result_list})
  with open(f"{experiment_code}.txt", "a") as f:
    f.write(str(result_log)+"\n")
    