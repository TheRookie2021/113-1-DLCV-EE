""" some class are put in here """
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
import csv
from PIL import Image 
# from torchvision.io import read_image
from torchvision.models import resnet50



# ***""""""""""""""""""""""""""""""""""""""""" Dataset """"""""""""""""""""""""""""""""""""""""""""""""""""

# class Mini_ImageNet_Augment_Rotate(Dataset):
#     def __init__(self, root, transform=None):
#         """ Intialize the dataset """
#         self.root = root # pretrain_dataset_root='../hw1_data/p1_data/mini/train'
#         self.ssl_rotate= { 0:0, 1:90, 2:180, 3:270}
#         self.filenames = []
#         self.transform = transforms.Compose([
#             transforms.Resize(128),
#             transforms.CenterCrop(128),
#             transforms.ToTensor(),
#             transforms.Normalize( mean=[0.485, 0.456, 0.406],
#                                   std =[0.229,0.224,0.225] )
#         ]) # by hw1 constrain
        
#         # 1. read filenames 
#         self.filenames = os.listdir(root)
#         self.len = len(self.filenames)

#     def __getitem__(self, index):
#         """ Get a sample from the dataset """
#         image_path = self.filenames[index]
#         image = Image.open(os.path.join(self.root, image_path))
        
#         # TODO:? self-supervise part (dataset has no label)
#         # data augmentation
#         label=random.randrange(0,4)
#         image=image.rotate(self.ssl_rotate[label])
#         if self.transform is not None:
#             image = self.transform(image)
#         return image, label

#     def __len__(self):
#         """ Total number of samples in the dataset """
#         return self.len
    
class Mini_ImageNet_unlabelled(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the dataset """
        self.root = root # pretrain_dataset_root='../hw1_data/p1_data/mini/train'
        self.ssl_rotate= { 0:0, 1:90, 2:180, 3:270}
        self.filenames = []
        self.transform = transforms.Compose([
            transforms.Resize(128),                            
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.TrivialAugmentWide(),

            transforms.ToTensor(),
            transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                  std =[0.229,0.224,0.225] )
        ]) # by hw1 constrain
        
        # 1. read filenames 
        self.filenames = os.listdir(root)
        self.len = len(self.filenames)

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_path = self.filenames[index]
        image = Image.open(os.path.join(self.root, image_path))
        
        # use BYOL no need for augmentation
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

class Office_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        
        self.img_labels = []
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                            transforms.Resize(128),                            
                            transforms.RandomCrop(128),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.TrivialAugmentWide(),
                            transforms.ToTensor(),
                            transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                            std =[0.229,0.224,0.225] )
                        ]) # by hw1 constrain

        # process .csv without using pandas
        with open(annotations_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                cols=row[0].split(',')
                self.img_labels.append([cols[1],cols[2]])
            self.img_labels=self.img_labels[1:]

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        img = Image.open(img_path)
        label = int(self.img_labels[idx][1])
        
        if self.transform:
            img = self.transform(img)

        return img, label
    
    def __len__(self):
        return len(self.img_labels)

class Test_Dataset(Dataset):
    """ use for infernce data """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        
        self.img_labels = []
        self.img_dir = img_dir
        self.transform = transform
        # process .csv without using pandas
        with open(annotations_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                cols=row[0].split(',')
                self.img_labels.append([cols[1],cols[2]])
            self.img_labels=self.img_labels[1:]

    def __getitem__(self, idx):
        filename=self.img_labels[idx][0]
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        img = Image.open(img_path)
        label = int(self.img_labels[idx][1])
        
        if self.transform:
            img = self.transform(img)

        return img,  label, filename
    
    def __len__(self):
        return len(self.img_labels)
# ***""""""""""""""""""""""""""""""""""""""""" NN Models """"""""""""""""""""""""""""""""""""""""""""""""""""
# class MyResNet(nn.Module):
#     def __init__(self):
#         super(MyResNet,self).__init__()
#         # see: https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
#         # self.backbone=nn.Sequential(*(list(resnet50(weights=None).children())[:-1])) # remove the last fc layer ((fc): Linear(in_features=2048, out_features=1000, bias=True))
#         # self.backbone=nn.Sequential(resnet50(weights=None)) # remove the last fc layer ((fc): Linear(in_features=2048, out_features=1000, bias=True))
#         # self.fc1 =nn.Sequential(
#         #     nn.Linear(in_features=2048, out_features=1024),
#         #     nn.Dropout(0.5)
#         # )
#         # self.fc2=nn.Sequential(
#         #     nn.Linear(in_features=1024, out_features=256),
#         #     nn.Dropout(0.5)
#         # )
#         # self.fc3 = nn.Linear(256, 4) 
        
#     def forward(self,x):
#         # x = self.backbone(x)
#         # print('Tensor size and type after backbone:', x.shape, x.dtype)
#         # x = x.view(x.size(0), -1)
#         # # print('Tensor size and type after x.view:', x.shape, x.dtype)
#         # x = self.fc1(x)
#         # # print('Tensor size and type after fc1:', x.shape, x.dtype)
#         # x = self.fc2(x)
#         # # print('Tensor size and type after fc2:', x.shape, x.dtype)
#         return x

class FinetuneResNet(nn.Module):
    def __init__(self):
        super(FinetuneResNet,self).__init__()
        # see: https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch
        # self.backbone=nn.Sequential(*(list(resnet50(weights=None).children())[:-1])) # remove the last fc layer ((fc): Linear(in_features=2048, out_features=1000, bias=True))
        self.backbone=resnet50(weights=None) # remove the last fc layer ((fc): Linear(in_features=2048, out_features=1000, bias=True))
        self.backbone.fc=nn.Identity() # simply forward the input of fc layer to output
        self.conv1 =nn.Sequential(
            nn.Conv1d(2048, 256, 1, stride=1),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc3 = nn.Linear(256, 65) 
        # self.fc2 = nn.Sequential(
        #     nn.Linear(2048, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        # ) 

    def forward(self,x):
        x = self.backbone(x)
        # print('Tensor size and type after x.backbone:', x.shape, x.dtype)
        # print('Tensor size and type after unsqueeze:', x.shape, x.dtype)
        x = torch.unsqueeze(x,-1)
        x = self.conv1(x)
        # print('Tensor size and type after x.conv:', x.shape, x.dtype)
        x = x.view(x.size(0), -1) # to flatten and fit the in_feature size of fc layers
        # x = self.fc2(x)
        x = self.fc3(x)
        return x