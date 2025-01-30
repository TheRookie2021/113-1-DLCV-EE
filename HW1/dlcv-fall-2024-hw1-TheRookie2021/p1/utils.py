""" some common function are put in here """
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader, random_split
import glob
import os
import csv
import random
import numpy as np
from tqdm import tqdm
# from torchvision.io import read_image
from PIL import Image
import argparse
from byol_pytorch import BYOL

MAX_NUM_SAVED_CKP=5

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='HW1 deep learning network.')
    parser.add_argument(
        '--experiment_code', type=str, default='', help='')
    parser.add_argument(
        '--note', type=str, help='')
    parser.add_argument(
        '--pretrain_from_scratch', default=False, type=bool, help='')
    parser.add_argument(
        '--inference_model', default='', type=str, help='')
    
    # TODO
    parser.add_argument(
        '--freeze_backbone', default=False, type=bool, help='')
    parser.add_argument(
        '--restart_finetune', default='', type=str, help='')
    
    parser.add_argument(
        '--pretrain_path', default='', type=str, help='can be .pth or the folder that save checkpoints')

    parser.add_argument(
        '--epoch_pretrain', default=1, type=int, help='')
    parser.add_argument(
        '--epoch_finetune', default=100, type=int, help='')
    parser.add_argument(
        '--train_batch_size', default=32, type=int, help='')
    parser.add_argument(
        '--test_batch_size', default=128, type=int, help='')
    
    parser.add_argument(
        '--lr', default=0.001, type=float, help='')
    parser.add_argument(
        '--weight_decay', default=0.0001, type=float, help='')
    parser.add_argument(
        '--save_interval', default=10000, type=int, help='')
    parser.add_argument(
        '--save_begin', default=0.5, type=float, help='')
    
    parser.add_argument(
        '--pretrain_dataset_root', default='./hw1_data/p1_data/mini/train', type=str, help='')
    parser.add_argument(
        '--fintune_dataset_root', default='./hw1_data/p1_data/office', type=str, help='')
    
    parser.add_argument(
        '--save_dir_pretrain_model', default='./model_checkpoints/p1/BYOL', type=str, help='')
    parser.add_argument(
        '--save_dir_finetune_model', default='./model_checkpoints/p1/default/finetune', type=str, help='')
    parser.add_argument(
        '--save_dir_result', default='./p1/record', type=str, help='')

    parser.add_argument(
        '--seed', default=58, type=int, help='')
    
    return parser.parse_args()



""""In order to do pretrain and finetune in one script"""
def find_best_model(dir_path, target='loss'):
    models=os.listdir(dir_path)
    best_indx=0
    if(target == 'acc'):
        best_acc=0
        for indx, name in enumerate(models):
            split_name= name.split('_')
            split_name= split_name[-1][:-4]
            acc=float(split_name[3:])
            print(acc)
            if(best_acc < acc):
                best_acc=acc
                best_indx=indx
        return os.path.join(dir_path, models[best_indx]) 


    best_loss=float("inf")
    for indx, name in enumerate(models):
        split_name= name.split('_')
        split_name= split_name[1].split('.')
        loss=int(split_name[0][1:])
        # print(name.split('_d'))
        # print(loss)
        if(best_loss>loss):
            best_loss=loss
            best_indx=indx

    return os.path.join(dir_path,models[best_indx]) 

def find_worst_model(dir_path, target='loss'):
    models=os.listdir(dir_path)
    worst_indx=0
    if(target == 'acc'):
        worst_acc=float("inf")
        for indx, name in enumerate(models):
            split_name= name.split('_')
            split_name= split_name[-1][:-4]
            acc=float(split_name[3:])
            print(acc)
            if(worst_acc > acc):
                worst_acc=acc
                worst_indx=indx
        return worst_acc, os.path.join(dir_path, models[worst_indx]) 

    worst_loss=0
    for indx, name in enumerate(models):
        split_name= name.split('_')
        split_name= split_name[1].split('.')
        loss=int(split_name[0][1:])

        print(loss)
        if(worst_loss<loss):
            worst_loss=loss
            worst_indx=indx

    return worst_loss, os.path.join(dir_path,models[worst_indx]) 

""" used for recording the log of each epoch """
def save_and_print(target, str):
    print(str)
    # TODO: decide the format
    # 1. str as final list of str only used once

def save_checkpoint(model, optimizer, save_dir, name ):
    # check if dir exist
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    checkpoint_path=os.path.join(save_dir, name)
    # save state_dict
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print(f'model saved to {checkpoint_path}')

# ============================================================================================================
def test(model, device, testset_loader, result_log):
    print("=== Test Phase ===")
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    model.eval()  # Important: set evaluation mode
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in tqdm(testset_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testset_loader.dataset)
    result_log.append('Test set: Average loss: {:.8f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(testset_loader.dataset),
        100. * correct / len(testset_loader.dataset)))
    print(result_log[-1]) 
    return test_loss, 100. * correct / len(testset_loader.dataset)

# ============================================================================================================
def train_save(model, device, epoch, lr, weight_decay, train_dataloader, test_dataloader , save_interval=1, save_begin=0.3, save_dir='', result_log=[] ):
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # TODO: test parameters, add scheduler, add early stop, log
    
    best_loss= 1
    best_test_acc= 0
    model.train()  # set training mode
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    train_acc_list=[]
    test_acc_list=[]
    
    for ep, _ in enumerate(range(epoch)):
        print(f"=== Train Phase, Epoch {ep}/{epoch} ===")
        total_loss=0
        correct=0
        model.train()  # set training mode
        for ind, (data, label) in enumerate(tqdm(train_dataloader)):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            total_loss+=float(loss)
            correct += pred.eq(label.view_as(pred)).sum().item()
            
        # scheduler.step()

        """deal with loss and acc output log"""
        avg_loss = total_loss/ len(train_dataloader.dataset)
        best_loss = min(best_loss, avg_loss)
        result_log.append('Train set: Average loss: {:.8f}, Accuracy: {}/{} ({:.0f}%)'.format(
                avg_loss, correct, 
                len(train_dataloader.dataset),
                100. * correct / len(train_dataloader.dataset)))
        print(result_log[-1])

        # step: validation
        test_loss, test_acc= test(model, device, test_dataloader,result_log)

        best_test_acc=max(best_test_acc,test_acc)
        train_acc_list.append(100. * correct / len(train_dataloader.dataset))
        test_acc_list.append(test_acc)
        if ep> ( epoch*save_begin) and ( ep%save_interval == 0 or best_test_acc == test_acc) :
            if os.path.isdir(save_dir):
                if(len(os.listdir(save_dir)) >= MAX_NUM_SAVED_CKP):
                    worst_acc, to_be_deleted=find_worst_model(save_dir, target='acc')
                    if worst_acc < test_acc:
                        result_log.append(f"out of preset buffer, delete worst model {to_be_deleted}")
                        print(result_log[-1])
                        os.remove(to_be_deleted)
                        save_checkpoint(model, optimizer, save_dir ,f"epoch{ep}_d{str(int(10000*test_loss)).zfill(4)}_test_acc{test_acc}.pth")
                    else:
                        pass
                else:
                    save_checkpoint(model, optimizer, save_dir ,f"epoch{ep}_d{str(int(10000*test_loss)).zfill(4)}_test_acc{test_acc}.pth")
            else:
                save_checkpoint(model, optimizer, save_dir ,f"epoch{ep}_d{str(int(10000*test_loss)).zfill(4)}_test_acc{test_acc}.pth")
    result_log.append([train_acc_list, test_acc_list])

# ============================================================================================================
def BYOL_train_save(model, device, epoch, lr, train_dataloader , save_interval=1, save_dir='',result_log=[] ):
    learner = BYOL(
        model,
        image_size = 128,
        hidden_layer = 'avgpool',
        use_momentum = False       # turn off momentum in the target encoder

    )
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(learner.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    # TODO: test parameters, add scheduler, add early stop, log

    model.train()  # set training mode
    best_loss= 1
    
    for ep, _ in enumerate(range(epoch)):
        print(f"=== Train Phase, Epoch {ep}/{epoch} ===")
        total_loss=0
        for ind, data in enumerate(tqdm(train_dataloader)):
            data = data.to(device)

            # BYOL method
            loss = learner(data)
            total_loss+=loss

            # calculate gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # BYOL method: update moving average of target encoder
            # learner.update_moving_average()
        # scheduler.step()

        avg_loss = total_loss/ len(train_dataloader)
        best_loss = min(best_loss, avg_loss) 
        result_log.append(f"Epoch {ep}\nAvg loss: {avg_loss}")
        print(result_log[-1])

        if ep % save_interval == 0 or avg_loss == best_loss :
            if os.path.isdir(save_dir):
                if(len(os.listdir(save_dir)) >= MAX_NUM_SAVED_CKP):
                    worst_loss, to_be_deleted=find_worst_model(save_dir, target='loss')
                    if worst_loss > avg_loss:
                        result_log.append(f"out of preset buffer, delete worst model {to_be_deleted}")
                        print(result_log[-1])
                        os.remove(to_be_deleted)
            save_checkpoint(model, optimizer, save_dir ,f"epoch{ep}_d{str(int(10000*avg_loss)).zfill(4)}.pth")


