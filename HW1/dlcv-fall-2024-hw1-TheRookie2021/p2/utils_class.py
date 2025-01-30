""" some class are put in here """
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
from PIL import Image 
import imageio
from torchvision.models import vgg16, VGG16_Weights 
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
# from tqdm import tqdm
# import torchvision
# from torchvision.io import read_image
# from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

class FocalLoss(nn.Module):
    """
    Multi-class Focal loss implementation
    paper: https://arxiv.org/abs/1708.02002\n
    github: https://github.com/ashawkey/FocalLoss.pytorch
    """
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.CE = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        Formula:
            CE(p,y) = CE(pt) = -log(pt)
            FL(pt) = -((1-pt)**gamma) log(pt).
        """
        log_pt = -self.CE(input, target)
        pt = torch.exp(log_pt)
        FL_pt = -0.25* ((1-pt)**self.gamma) * log_pt 
        return FL_pt    

# ***""""""""""""""""""""""""""""""""""""""""" Dataset """"""""""""""""""""""""""""""""""""""""""""""""""""
class Semantic_Segmentation_Dataset(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the dataset """
        self.root = root # pretrain_dataset_root='../hw1_data/p1_data/mini/train'
        self.filenames = []
        self.transform =  transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(   mean=[0.485, 0.456, 0.406],
                                                    std =[0.229,0.224,0.225] )  ]) 
        # 1. read filenames 
        sat_filenames = [filename for filename in sorted(os.listdir(root)) if "sat" in filename]
        masks =self.read_masks(root)
        self.target_label_pairs=[(x,y) for x,y in zip(sat_filenames, masks)]
        self.len = len(self.target_label_pairs)
        
        # !-->>>> TOO slow
        # self.dict={ 
        #     (0,255,255):0,  # ubuan
        #     (255,255,0):1,  # agriculture
        #     (255,0,255):2,  # rangeland
        #     (0,255,0):3,    # forest
        #     (0,0,255):4,    # water
        #     (255,255,255):5,# barren
        #     (0,0,0):6,      # unknown
        # }
    def read_masks(self,filepath):
        '''
        Read masks from directory and tranform to categorical
        '''
        file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
        file_list.sort()
        n_masks = len(file_list)
        masks = np.empty((n_masks, 512, 512))
        # print(len(file_list))
        for i, file in enumerate(file_list):
            mask = imageio.imread(os.path.join(filepath, file))
            mask = (mask >= 128).astype(int)
            mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
            masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
            masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
            masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
            masks[i, mask == 2] = 3  # (Green: 010) Forest land 
            masks[i, mask == 1] = 4  # (Blue: 001) Water 
            masks[i, mask == 7] = 5  # (White: 111) Barren land 
            masks[i, mask == 0] = 6  # (Black: 000) Unknown 

        return masks

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        target_label = self.target_label_pairs[index]
        target = Image.open(os.path.join(self.root, target_label[0]))
        mask = torch.tensor(target_label[1], dtype=torch.long)
        # RGB_mask = Image.open(os.path.join(self.root, target_label[1]))

        if self.transform is not None:
            target = self.transform(target)
        # label = self.RGB_to_label(RGB_mask)
        return target, mask

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
    # !-->>>> TOO slow
    # def RGB_to_label(self, img): 
    #     # print(self.dict[img.getpixel((0,0))])
    #     labeling=np.ones((img.size[0],img.size[1]),dtype=int)
    #     for i in range(img.size[0]):
    #         for j in range(img.size[1]):
    #             labeling[i,j]= self.dict[img.getpixel((i,j))] 
    #     return torch.tensor(labeling,dtype=torch.long)

# class Test_Semantic_Segmentation_Dataset(Dataset):
#     def __init__(self, root, transform=None):
#         """ Intialize the dataset """
#         self.root = root # pretrain_dataset_root='../hw1_data/p1_data/mini/train'
#         self.filenames = []
#         self.transform =  transforms.Compose([
#                             transforms.ToTensor(),
#                             transforms.Normalize(   mean=[0.485, 0.456, 0.406],
#                                                     std =[0.229,0.224,0.225] )  ]) 
#         # 1. read filenames 
#         sat_filenames = [filename for filename in sorted(os.listdir(root)) if "sat" in filename]
#         masks =self.read_masks(root)
#         self.target_label_pairs=[(x,y) for x,y in zip(sat_filenames, masks)]
#         self.len = len(self.target_label_pairs)
        
#     def read_masks(self,filepath):
#         '''
#         Read masks from directory and tranform to categorical
#         '''
#         file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
#         file_list.sort()
#         n_masks = len(file_list)
#         masks = np.empty((n_masks, 512, 512))
#         # print(len(file_list))
#         for i, file in enumerate(file_list):
#             mask = imageio.imread(os.path.join(filepath, file))
#             mask = (mask >= 128).astype(int)
#             mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
#             masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
#             masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
#             masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
#             masks[i, mask == 2] = 3  # (Green: 010) Forest land 
#             masks[i, mask == 1] = 4  # (Blue: 001) Water 
#             masks[i, mask == 7] = 5  # (White: 111) Barren land 
#             masks[i, mask == 0] = 6  # (Black: 000) Unknown 

#         return masks

#     def __getitem__(self, index):
#         """ Get a sample from the dataset """
#         target_label = self.target_label_pairs[index]
#         filename=target_label[0]
#         target = Image.open(os.path.join(self.root, target_label[0]))
#         mask = torch.tensor(target_label[1], dtype=torch.long)
#         # RGB_mask = Image.open(os.path.join(self.root, target_label[1]))

#         if self.transform is not None:
#             target = self.transform(target)
#         # label = self.RGB_to_label(RGB_mask)
#         return target, mask, filename

#     def __len__(self):
#         """ Total number of samples in the dataset """
#         return self.len


    
# ***""""""""""""""""""""""""""""""""""""""""" NN Models """""""""""""""""""""""""""""""""""""""""""""""""""" 
class VGG16_FCN32s(nn.Module):
    def __init__(self, N_class=7):
        # ==== Loss: pixel-wise cross-entropy (compute cross-entropy between each pixel and its label, and average over all of them)
        super(VGG16_FCN32s,self).__init__()
        # see: https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
        # self.backbone=nn.Sequential(*(list(resnet50(weights=None).children())[:-1])) # remove the last fc layer ((fc): Linear(in_features=2048, out_features=1000, bias=True))
        # note: only need the vgg16.features layers( equals FCNs conv1~5)
        self.vgg16=vgg16(weights=VGG16_Weights.DEFAULT).features 
        
        # note conv6~7: conv, 4096 chennel, upsampling
        self.conv6_1 = nn.Sequential( nn.Conv2d(512, 4096, 3, padding=1), nn.ReLU(), nn.Dropout2d())
        self.conv7_1= nn.Sequential( nn.Conv2d(4096, 4096, 3, padding=1), nn.ReLU(), nn.Dropout2d())

        # 1*1 convolution with channel dimension C to predict scores: upsampling in one step after using  1*1 conv
        self.score_pred = nn.Conv2d(4096, N_class, 1)
        self.upsample = nn.ConvTranspose2d(N_class, N_class, 32, stride=32)

    def forward(self,x):
        # print('Tensor size and type:', x.shape, x.dtype)
        x = self.vgg16(x)
        # print('Tensor size and type after VGG16 features:', x.shape, x.dtype)
        x = self.conv6_1(x)
        # print('Tensor size and type conv6_1 :', x.shape, x.dtype)
        x = self.conv7_1(x)
        # print('Tensor size and type conv7_1 :', x.shape, x.dtype)
        x = self.score_pred(x)
        # print('Tensor size and type score_pred :', x.shape, x.dtype)
        x = self.upsample(x)
        # print('Tensor size and type upsample :', x.shape, x.dtype)
        return x
    
class LRASPP_MobileNet_V3(nn.Module):
    """
    Source code:
    https://pytorch.org/vision/0.12/_modules/torchvision/models/segmentation/lraspp.html
    """
    def __init__(self, N_class=7):
        # ==== Loss: pixel-wise cross-entropy (compute cross-entropy between each pixel and its label, and average over all of them)
        super(LRASPP_MobileNet_V3, self).__init__()
        
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        used_model = lraspp_mobilenet_v3_large(weights=weights)
        # see: https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
        # self.backbone=nn.Sequential(*(list(resnet50(weights=None).children())[:-1])) # remove the last fc layer ((fc): Linear(in_features=2048, out_features=1000, bias=True))
        # note: only need the vgg16.features layers( equals FCNs conv1~5)
        self.backbone=used_model.backbone
        
        self.cbr=used_model.classifier.cbr
        self.scale=used_model.classifier.scale

        self.low_classifier=nn.Conv2d(40, N_class, kernel_size=(1, 1), stride=(1, 1))
        self.high_classifier=nn.Conv2d(128, N_class, kernel_size=(1, 1), stride=(1, 1))
        # FCN: one step upsampling stride
        # self.upsample = nn.ConvTranspose2d(21, N_class, 8, stride=8)
        
    def forward(self, intput, ratio=0.8):
        feature = self.backbone(intput)

        low = feature["low"]
        high = feature["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s

        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        out= ratio*self.low_classifier(low) + (1-ratio)*self.high_classifier(x)
        out = F.interpolate(out, size=intput.shape[-2:], mode="bilinear", align_corners=False)
        # out = self.upsample(out)
        return out
    
# class LRASPP_MobileNet_V3_Custom(nn.Module):
#     """
#     Source code:
#     https://pytorch.org/vision/0.12/_modules/torchvision/models/segmentation/lraspp.html
#     """
#     def __init__(self, N_class=7):
#         # ==== Loss: pixel-wise cross-entropy (compute cross-entropy between each pixel and its label, and average over all of them)
#         super(LRASPP_MobileNet_V3_Custom, self).__init__()
        
#         weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
#         used_model = lraspp_mobilenet_v3_large(weights=weights)
#         # see: https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
#         # self.backbone=nn.Sequential(*(list(resnet50(weights=None).children())[:-1])) # remove the last fc layer ((fc): Linear(in_features=2048, out_features=1000, bias=True))
#         # note: only need the vgg16.features layers( equals FCNs conv1~5)
#         self.backbone=used_model.backbone
        
#         self.cbr=used_model.classifier.cbr
#         self.scale=used_model.classifier.scale

#         self.low_classifier=nn.Conv2d(40, 21, kernel_size=(1, 1), stride=(1, 1))
#         self.high_classifier=nn.Conv2d(128, 21, kernel_size=(1, 1), stride=(1, 1))
#         # FCN: one step upsampling stride
#         self.upsample = nn.ConvTranspose2d(21, N_class, 8, stride=8)
        
#     def forward(self, intput, ratio=0.3):
#         feature = self.backbone(intput)

#         low = feature["low"]
#         high = feature["high"]

#         x = self.cbr(high)
#         s = self.scale(high)
#         x = x * s

#         x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
#         out= ratio*self.low_classifier(low) + (1-ratio)*self.high_classifier(x)
#         # out = F.interpolate(out, size=intput.shape[-2:], mode="bilinear", align_corners=False)
#         out = self.upsample(out)
#         return out
 
# class LRASPP_MobileNet_V3P3(nn.Module):
#     """
#     Source code:
#     https://pytorch.org/vision/0.12/_modules/torchvision/models/segmentation/lraspp.html
#     """
#     def __init__(self, N_class=7):
#         # ==== Loss: pixel-wise cross-entropy (compute cross-entropy between each pixel and its label, and average over all of them)
#         super(LRASPP_MobileNet_V3P3, self).__init__()
#         weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
#         used_model = lraspp_mobilenet_v3_large(weights=weights)
        
#         self.backbone=used_model.backbone
        
#         self.cbr1=used_model.classifier.cbr
#         self.scale1=used_model.classifier.scale
#         self.low_classifier1=nn.Conv2d(40, 21, kernel_size=(1, 1), stride=(1, 1))
#         self.high_classifier1=nn.Conv2d(128, 21, kernel_size=(1, 1), stride=(1, 1))
#         self.upsample1 = nn.ConvTranspose2d(21, N_class, 8, stride=8)
        
#         self.cbr2=used_model.classifier.cbr
#         self.scale2=used_model.classifier.scale
#         self.low_classifier2=nn.Conv2d(40, N_class, kernel_size=(1, 1), stride=(1, 1))
#         self.high_classifier2=nn.Conv2d(128, N_class, kernel_size=(1, 1), stride=(1, 1))
#         # self.upsample2 = nn.ConvTranspose2d(21, N_class, 8, stride=8)
        
#         # FCN: one step upsampling stride
#     def forward(self, intput):
#         feature = self.backbone(intput)

#         low = feature["low"]
#         high = feature["high"]

#         x1 = self.cbr1(high)
#         s1 = self.scale1(high)
#         x1 = x1 * s1
#         x1 = F.interpolate(x1, size=low.shape[-2:], mode="bilinear", align_corners=False)
#         out1= self.low_classifier1(low) +  self.high_classifier1(x1)
#         out1 = self.upsample1(out1)

#         x2 = self.cbr2(high)
#         s2 = self.scale2(high)
#         x2 = x2 * s2
#         x2 = F.interpolate(x2, size=low.shape[-2:], mode="bilinear", align_corners=False)
#         out2= self.low_classifier2(low) +  self.high_classifier2(x2)
#         out2 = F.interpolate(out2, size=intput.shape[-2:], mode="bilinear", align_corners=False)
#         # out2 = self.upsample(out2)

#         return out1+out2

# class Deeplabv3_Resnet101(nn.Module):
#     """
#     Source code:
#     https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html#torchvision.models.segmentation.DeepLabV3_ResNet101_Weights
#     """
#     def __init__(self, N_class=7):
#         # ==== Loss: pixel-wise cross-entropy (compute cross-entropy between each pixel and its label, and average over all of them)
#         super(Deeplabv3_Resnet101, self).__init__()
        
#         weights = DeepLabV3_ResNet101_Weights.DEFAULT
#         used_model = deeplabv3_resnet101(weights=weights)
#         used_model.classifier[-1]=nn.Conv2d(256, N_class, kernel_size=(1, 1), stride=(1, 1))
#         used_model.aux_classifier[-1]=nn.Conv2d(256, N_class, kernel_size=(1, 1), stride=(1, 1))
#         self.backbone=used_model 

#     def forward(self, x):
#         x=self.backbone(x)
#         return x["out"]


