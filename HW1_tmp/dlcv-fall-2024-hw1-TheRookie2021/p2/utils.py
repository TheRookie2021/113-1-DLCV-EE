""" some common function are put in here """
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import os
import numpy as np
from tqdm import tqdm
# from torchvision.io import read_image
from PIL import Image
import argparse
from utils_class import FocalLoss 

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
        '--inference_model', default='', type=str, help='')
    
    # TODO
    parser.add_argument(
        '--freeze_backbone', default=False, type=bool, help='')
    parser.add_argument(
        '--restart_finetune', default='', type=str, help='')

    parser.add_argument(
        '--epoch', default=100, type=int, help='')
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
        '--dataset_root', default='./hw1_data/p2_data', type=str, help='')
    
    parser.add_argument(
        '--save_dir_model', default='./model_checkpoints/p2', type=str, help='')
    parser.add_argument(
        '--save_dir_result', default='./p2/record', type=str, help='')

    parser.add_argument(
        '--seed', default=58, type=int, help='')
    
    return parser.parse_args()

# ==================================================for IO==========================================================
def find_best_model(dir_path, target='loss'):
    """
    to find the best model in the given folderaccording to loss, acc, mean iou (based on the .pth file name)
    """
    models=[model for model in os.listdir(dir_path) if model.endswith('.pth')]
    best_indx=0
    # acc, iou
    if(target.lower() == 'acc' or target.lower()=='iou'):
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
    # loss
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
    """
    to find the worst model in the given folderaccording to loss, acc, mean iou (based on the .pth file name)
    """
    models=[model for model in os.listdir(dir_path) if model.endswith('.pth')]
    worst_indx=0

    # acc, iou
    if(target.lower() == 'acc' or target.lower()=='iou'):
        worst_acc=float("inf")
        for indx, name in enumerate(models):
            split_name= name.split('_')
            split_name= split_name[-1][:-4]
            acc=float(split_name[3:])
            print(acc)
            if(worst_acc > acc):
                worst_acc=acc
                worst_indx=indx
        # print(worst_acc)
        return worst_acc, os.path.join(dir_path, models[worst_indx]) 
    # loss
    worst_loss=0
    for indx, name in enumerate(models):
        split_name= name.split('_')
        split_name= split_name[1].split('.')
        loss=int(split_name[0][1:])

        print(loss)
        if(worst_loss<loss):
            worst_loss=loss
            worst_indx=indx
    print(worst_loss)
    return worst_loss, os.path.join(dir_path,models[worst_indx]) 

# def save_and_print(target, str):
#     """ used for recording the log of each epoch """
#     print(str)
    
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

# def conver_RGB_to_label(RGB_tuple):
#     dict={
#         (0,255,255):0,  # ubuan
#         (255,255,0):1,  # agriculture
#         (255,0,255):2,  # rangeland
#         (0,255,0):3,    # forest
#         (0,0,255):4,    # water
#         (255,255,255):5,# barren
#         (0,0,0):6,      # unknown
#     }
#     return dict[RGB_tuple]

# def mean_iou_score_torch(pred, labels):
#     '''
#     Compute mean IoU score over 6 classes
#     '''
#     # print(pred.shape, pred.dtype)
#     # print(labels.shape, labels.dtype)
#     mean_iou = 0
#     for i in range(6):
#         # mask=torch.full(pred.size(), i,dtype=torch.int64).to("cuda")
#         # print(mask.shape,mask.dtype)
#         pred_mask=torch.eq(pred ,i).long()
#         gt_mask=torch.eq(labels ,i).long()
#         intersect=pred_mask * gt_mask

#         tp_fp = torch.sum(pred_mask, dtype=torch.int64)
#         tp_fn = torch.sum(gt_mask, dtype=torch.int64)
#         tp = torch.sum( intersect, dtype=torch.int64)

#         iou =  0 if (tp_fp==0 and tp_fn==0) else tp / ((tp_fp + tp_fn - tp))  # intersect/union
#         mean_iou += iou / 6
#         # print(f"tp_fp:{tp_fp}\ntp_fn:{tp_fn}\ntp:{tp}\niou:{iou}\n")
#         # print('class #%d : %1.5f'%(i, iou))
#     # print('\nmean_iou: %f\n' % mean_iou)
#     return mean_iou

# def test_torch(model, device, testset_loader, result_log):
#     print("=== Test Phase ===")
#     # criterion = nn.CrossEntropyLoss(reduction="mean")
#     criterion = FocalLoss()

#     test_loss = 0
#     mIoU=0
#     N=len(testset_loader)
#     model.eval()  # Important: set evaluation mode
#     with torch.no_grad(): # This will free the GPU memory used for back-prop
#         for data, target in tqdm(testset_loader):
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item() # sum up batch loss
#             # TODO: fix bug, in mean_iou_score_torch 
#             pred = torch.argmax(output, dim=1)
#             mIoU += mean_iou_score_torch(pred.long(), target.long())
#             del pred, target
#     mIoU/=N
#     test_loss /= len(testset_loader.dataset)
#     result_log.append('Test set: Average loss: {:.8f}, Mean IOU : {:.4f}%)'.format(test_loss, mIoU*100))
#     print(result_log[-1]) 
#     return test_loss, mIoU

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    # print(pred.shape, pred.dtype)
    # print(labels.shape, labels.dtype)
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        # print(f"tp_fp:{tp_fp}\ntp_fn:{tp_fn}\ntp:{tp}\niou:{iou}\n")
        # print('class #%d : %1.5f'%(i, iou))
    # print('\nmean_iou: %f\n' % mean_iou)
    return mean_iou
        
def test(model, device, testset_loader, result_log):
    print("=== Test Phase ===")
    criterion = nn.CrossEntropyLoss(reduction="mean")
    test_loss = 0
    all_preds=[]
    all_gt=[]

    model.eval()  # Important: set evaluation mode
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in tqdm(testset_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss

            pred = output.argmax(dim=1)
            pred= pred.detach().cpu().numpy().astype(np.int64)
            target = target.detach().cpu().numpy().astype(np.int64)
            all_preds.append(pred)
            all_gt.append(target)
            
    test_loss /= len(testset_loader.dataset)
    mIoU = mean_iou_score(np.concatenate(all_preds, axis=0), np.concatenate(all_gt, axis=0))
    result_log.append('Test set: Average loss: {:.8f}, Mean IOU : {:.4f}%)'.format(test_loss, mIoU*100))
    print(result_log[-1]) 
    return test_loss, mIoU

# ============================================================================================================
def train_save(model, device, epoch, lr, weight_decay, train_dataloader, test_dataloader , save_interval=1, save_begin=0.3, save_dir='', result_log=[] ):
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=epoch)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # criterion = nn.CrossEntropyLoss(reduction="mean")
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # criterion = FocalLoss()
    
    
    best_loss= 1
    best_test_iou= 0
    train_iou_list=[]
    test_iou_list=[]

    model.train()  # set training mode
    N=len(train_dataloader)
    for ep, _ in enumerate(range(epoch)):
        print(f"=== Train Phase, Epoch {ep}/{epoch} ===")
        total_loss=0
        mIoU=0
        model.train()  # set training mode
        # all_preds=None
        # all_gt=None
        for ind, (data, label) in enumerate(tqdm(train_dataloader)):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            # step: evaluation (loss and iou) 
            total_loss+=float(loss)        
            pred = output.argmax(dim=1) # compress 7 channel into a 2D 1 channel map
            # pred= pred.detach().cpu().numpy().astype(np.int64)
            # label = label.detach().cpu().numpy().astype(np.int64)
            # mIoU += mean_iou_score(pred, label)
            # TODO fix bug: mean_iou_score_torch 
            # mIoU += mean_iou_score_torch(pred, label)
            # if all_preds == None: all_preds=pred
            # else:    all_preds=torch.cat((all_preds,pred), 0)
            # if all_gt == None: all_gt=label
            # else: all_gt=torch.cat((all_gt, label), 0)
            # del pred, label
                
        # step: deal with output log, loss and mIOU
        mIoU/=N
        train_loss = total_loss/ len(train_dataloader.dataset)
        best_loss = min(best_loss, train_loss)
        result_log.append('Train set: Average loss: {:.8f}, Mean IOU : {:.4f}%)'.format(train_loss, mIoU*100))
        print(result_log[-1])

        # step: validation
        test_loss, test_iou= test(model, device, test_dataloader, result_log)
        best_test_iou=max(best_test_iou,test_iou)
        train_iou_list.append(mIoU)
        test_iou_list.append(test_iou)
        scheduler.step()
        
        # step: saving
        if ep in [0,int(epoch/2),epoch-1]:
            save_checkpoint(model, optimizer, os.path.join(save_dir,"required") ,f"epoch{ep}_d{str(int(10000*test_loss)).zfill(4)}_test_iou{test_iou*100:.4f}.pth")
                    
        if ep> ( epoch*save_begin) and ( ep%save_interval == 0 or best_test_iou == test_iou) :
            if os.path.isdir(save_dir) and (len(os.listdir(save_dir)) >= MAX_NUM_SAVED_CKP): # note: if dir exist, and it's out of buffer
                    worst_iou, to_be_deleted=find_worst_model(save_dir, target='iou')
                    print(f"worst_iou:{worst_iou}, test_iou:{test_iou *100},  worst model {to_be_deleted}")
                    if worst_iou < test_iou *100: # note : replace if the worst model is worst than current ckpt
                        result_log.append(f"out of preset buffer, delete worst model {to_be_deleted}")
                        print(result_log[-1])
                        os.remove(to_be_deleted)
                        save_checkpoint(model, optimizer, save_dir ,f"epoch{ep}_d{str(int(10000*test_loss)).zfill(4)}_test_iou{test_iou*100:.4f}.pth")
                    else:
                        print("do nothing")
            else:
                save_checkpoint(model, optimizer, save_dir ,f"epoch{ep}_d{str(int(10000*test_loss)).zfill(4)}_test_iou{test_iou*100:.4f}.pth")

    # end of epoch loop

    result_log.append([train_iou_list, test_iou_list])
    return "finish training"

# ============================================================================================================
