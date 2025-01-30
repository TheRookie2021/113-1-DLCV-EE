import torch
from torch.utils.data import Dataset
import torchvision.transforms  as transforms
from PIL import Image
import os

# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt 
def fileNamer(path):
# """
#     if not exist: original name of filepath
#     if exist:
#         append a number if there is no number
#         add 1 to existed number
# """    
    counter=1
    name, extension = os.path.splitext(path)
    while os.path.exists(path):
        path= name +f"({counter})"+ extension
        counter+=1
    return path


class CustomImageDataset(Dataset):
    def __init__(self, img_dir,     transform=transforms.Compose([transforms.Resize((480, 640)), transforms.ToTensor()]), target_transform=None):
        self.img_dir = img_dir
        self.filepath=[os.path.join(img_dir, name) for name in sorted(os.listdir(img_dir)) ]
        self.transform = transform
        # self.img_labels = json_file
        # self.target_transform = target_transform
        print(self.filepath[:5])
    def __len__(self):
        return len(self.filepath)

    def __getitem__(self, idx):
        img_path = self.filepath[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        
        # label = self.img_labels.iloc[idx, 1]
        # if self.target_transform:
        #     label = self.target_transform(label)

        return self.filepath[idx].split(".")[0], image
    