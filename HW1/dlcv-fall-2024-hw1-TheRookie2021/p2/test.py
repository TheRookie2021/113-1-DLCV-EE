""" some common function are put in here """
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split

import os
import numpy as np
from tqdm import tqdm
# from torchvision.io import read_image
from PIL import Image
from utils_class import VGG16_FCN32s, LRASPP_MobileNet_V3, Semantic_Segmentation_Dataset
# from utils import  test

def mean_iou_score_torch(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    print(pred.shape, pred.dtype)
    print(labels.shape, labels.dtype)
    mean_iou = 0
    for i in range(6):
        tp_fp = torch.sum((pred == i))
        tp_fn = torch.sum((labels == i))
        tp = torch.sum(((pred == i) * (labels == i)))
        iou = tp / ((tp_fp + tp_fn - tp)+1)
        mean_iou += iou / 6
        print(f"tp_fp:{tp_fp}\ntp_fn:{tp_fn}\ntp:{tp}\niou:{iou}\n")
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)
    return mean_iou

def test_torch(model, device, testset_loader, result_log):
    print("=== Test Phase ===")
    criterion = nn.CrossEntropyLoss(reduction="mean")
    test_loss = 0
    total_mean_IOU = 0
    all_preds=None
    all_gt=None
    mIoU=0
    model.eval()  # Important: set evaluation mode
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in tqdm(testset_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1)
            if all_preds == None: all_preds=pred
            else:    all_preds=torch.cat((all_preds,pred), 0)
            if all_gt == None: all_gt=target
            else: all_gt=torch.cat((all_gt, target), 0)
            
    test_loss /= len(testset_loader.dataset)
    mIoU = mean_iou_score_torch(all_preds,all_gt)
    result_log.append('Train set: Average loss: {:.8f}, Mean IOU : {:.4f}%)'.format(test_loss, mIoU*100))
    print(result_log[-1]) 
    return test_loss, mIoU

root=os.path.join("./hw1_data/p2_data","validation")
test_dataset=Semantic_Segmentation_Dataset(root=os.path.join("./hw1_data/p2_data","validation"))
print(len(test_dataset))
print(test_dataset[0][0].shape)
print(test_dataset[0][1].shape)

    
# print(read_masks(root).shape)
# print(read_masks(root)[5])
test_dataloader=DataLoader(test_dataset, batch_size=8, shuffle=True)
# # print(test_dataset[0][0].shape, test_dataset[0][0].dtype)
# # print(test_dataset[0][1].shape, test_dataset[0][1].dtype)
# # pred=model(test_dataset[10][0].to('cuda'))
# # label=test_dataset[10][1].to('cuda')
# # print(pred.shape, pred.dtype)
# # print(label.shape, label.dtype)
# # print(mean_iou_score(pred=label,labels=label))
dir="./model_checkpoints/p2/model_B/model_B01"
from utils import find_worst_model
worst_iou, to_be_deleted=find_worst_model(dir, target='iou')
print(f"worst_iou:{worst_iou}, test_iou:{100},  worst model {to_be_deleted}")
if worst_iou < 100: # note : replace if the worst model is worst than current ckpt
    print(f"out of preset buffer, delete worst model {to_be_deleted}")
    os.remove(to_be_deleted)
    print("removed")

else:
    print("do nothing")
            # model=LRASPP_MobileNet_V3(7).to("cuda")
# path=os.listdir(dir)
# for p in path:
#     print(f"loading {p}")
#     model.load_state_dict(torch.load(os.path.join(dir,p), weights_only=True)['state_dict'])
#     print(test_torch(model,"cuda",test_dataloader,[]))
#     print(test(model,"cuda",test_dataloader,[]))
#     break
# def pixelwise_CrossEntropyLossA(output, label):
#     # torch.Size([Batch_size, 7, 512, 512]) torch.float32
#     print(output.shape)
#     print(label.shape)
#     criterion = nn.CrossEntropyLoss(reduction="mean")
#     loss_mean=criterion(output,label)

#     criterion = nn.CrossEntropyLoss(reduction="sum")
#     loss_sum=criterion(output,label)
    
#     #         break
#     #     break
#     return loss_mean, loss_sum

# def pixelwise_CrossEntropyLossB(output, label):
#     # torch.Size([Batch_size, 7, 512, 512]) torch.float32
#     criterion = nn.CrossEntropyLoss()
#     print(output.shape)
#     print(label.shape)
#     batch_size=output.shape[0]
#     channel=output.shape[1]
#     width=output.shape[2]
#     height=output.shape[3]
    
#     total_loss=0
#     for i in range(width):
#         for j in range(height):
#             # print(output[:,:,i,j].shape,label[:,i,j].shape)
#             # print(output[:,:,i,j].dtype,label[:,i,j].dtype)
#             total_loss+=criterion(output[:,:,i,j],label[:,i,j])
#     #         break
#     #     break
#     return total_loss/(batch_size*width*height)

# output = torch.randn(4, 7, 512, 512).to('cuda')
# label = torch.ones(4, 512, 512,dtype=torch.long).to('cuda')

# print(pixelwise_CrossEntropyLossA(output,label))
# print(pixelwise_CrossEntropyLossB(output,label)) # god dame slow, is it correct?
# # tensor(0.5864, device='cuda:0')