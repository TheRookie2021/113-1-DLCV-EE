""" some class are put in here """
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ContextUnet import ContextUnet 
from tqdm import tqdm
from PIL import Image

import csv
MAX_SAVE_BUFFER=5
MNISTM_MEAN = [0.5, 0.5, 0.5]
MNISTM_STD = [0.5, 0.5, 0.5]
SVHN_MEAN = [0.5, 0.5, 0.5]
SVHN_STD = [0.5, 0.5, 0.5]

TRANSFORM_MNISTM= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNISTM_MEAN, MNISTM_STD)
            # lambda x: 2*x - 1       # normalized to range [-1,1]
            # transforms.Normalize( mean=[0.485, 0.456, 0.406],
            #                       std =[0.229,0.224,0.225] )
        ]) 
TRANSFORM_SVHN= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD)
            # lambda x: 2*x - 1       # normalized to range [-1,1]
            # transforms.Normalize( mean=[0.485, 0.456, 0.406],
            #                       std =[0.229,0.224,0.225] )
        ]) 

# ================================================================def========================================================================
# note: other tools, saving, finding...

def torch_to_PIL(img, dataset_id=None):
    # new_img=img.clone().detach()
    new_img=transforms.functional.to_pil_image(img)
    # if dataset_id!=0:
    #     new_img=transforms.functional.adjust_brightness(new_img,1.2) # for some reason, the output seems transparent
    #     new_img=transforms.functional.adjust_contrast(new_img,1.5) # for some reason, the output seems transparent
    # else:
    #     new_img=transforms.functional.adjust_contrast(new_img,1.8) # for some reason, the output seems transparent
        
    # print(new_img.mode)
    # img = img.detach().cpu().numpy()
    # img = Image.fromarray(img.astype('uint8'), 'RGB')
    # img.save(os.path.join(folder, str(index))+'.jpg')
    return new_img

def find_model(dir_path, metric='loss', target=['worst','best']):
    """
    to find the worst model in the given folderaccording to loss, acc, mean iou (based on the .pth file name)
    """
    if len(os.listdir(dir_path))==0: return
    models=[model for model in os.listdir(dir_path) if model.endswith('.pth')]
    # loss
    if target=='worst':
        worst_indx=0
        worst_loss=0
        for indx, name in enumerate(models):
            split_name= name.split('_')
            # split_name= split_name[1].split('.')
            loss=float(split_name[1])
            if(worst_loss<loss):
                worst_loss=loss
                worst_indx=indx
        print(f"{worst_loss=}")
        return worst_loss, os.path.join(dir_path,models[worst_indx]) 
    
    if target=="best":
        best_indx=0
        best_loss=float('inf')
        for indx, name in enumerate(models):
            split_name= name.split('_')
            # split_name= split_name[1].split('.')
            loss=float(split_name[1])
            if(best_loss>loss):
                best_loss=loss
                best_indx=indx
        print(f"{best_loss=}")
        return best_loss, os.path.join(dir_path,models[best_indx]) 


# note: Dataset
class MNISTN_Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        
        self.img_labels = []
        self.img_dir = img_dir
        self.transform = transform
        # process .csv without using pandas
        with open(annotations_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                cols=row[0].split(',')
                self.img_labels.append([cols[0],cols[1]]) # img_name, img_label
            self.img_labels=self.img_labels[1:] # trim the header row

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        img = Image.open(img_path)
        label = int(self.img_labels[idx][1])
        
        if self.transform!=None:
            img = self.transform(img)

        return img, label
    
    def __len__(self):
        return len(self.img_labels)

class SVHN_Dataset(Dataset):
    """
    modify the definition of label by shifting 10 (adding 10) in order to combine with MNISTN while make the two seperable
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        
        self.img_labels = []
        self.img_dir = img_dir
        self.transform = transform
        # process .csv without using pandas
        with open(annotations_file, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in spamreader:
                cols=row[0].split(',')
                self.img_labels.append([cols[0],cols[1]]) # img_name, img_label
            self.img_labels=self.img_labels[1:] # trim the header row

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        img = Image.open(img_path)
        # 
        label = int(self.img_labels[idx][1])+10
        
        if self.transform !=None:
            img = self.transform(img)

        return img, label
    
    def __len__(self):
        return len(self.img_labels)
    
# note: DDPM algorithm
# Forward pass:  gradually transforms the image into a Normal Distribution by adding noise over T time steps.
# Reverse pass:  gradually denoise the noisy image xT ~ N(0, 1) in T time steps.
class DDPM_forward_pass:
    """
    no trainable parameter in this phase
    1. adds the noise gradually according to the Gaussian Distribution
    2. Reparameterization Trick
    """
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print(self.device)
        # para for gaussian noise
        self.betas = torch.linspace(start=beta_start, end=beta_end, steps=int(T+1))
        # para for reparameterization trick: x_t= (alpha**1/2)x_0+((1-alpha)**1/2)*eps
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_1_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
    
    def add_noise(self, input_img, noise, t):
        # transform data
        # print(t.shape, t)
        sqrt_alpha_bar_t = self.sqrt_alpha_bars.to(self.device)[t]
        sqrt_1_minus_alpha_bars = self.sqrt_1_minus_alpha_bars.to(self.device)[t]
        sqrt_alpha_bar_t = sqrt_alpha_bar_t[:, None, None, None] # to fit input_img dimension [B,C,H,W]
        sqrt_1_minus_alpha_bars = sqrt_1_minus_alpha_bars[:, None, None, None] # to fit input_img dimension [B,C,H,W]
        
        # reparameterization trick: x_t= (alpha**1/2)x_0+((1-alpha)**1/2)*eps
        return (sqrt_alpha_bar_t * input_img) + (sqrt_1_minus_alpha_bars * noise)        

class DDPM_reverse_pass:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.betas = torch.linspace(start=beta_start, end=beta_end, steps=T)
        
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_1_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        # print(self.betas.shape)
        # print(self.alpha_bars.shape)
        # print(self.sqrt_1_minus_alpha_bars.shape)
        # print(self.alpha_bars[-10:])
        # print(self.sqrt_alpha_bars[-10:])

        
    def sample_timestep_prev(self, xt, noise_pred, t):
        """
        Algo2-4
        """
        # move to gpu
        t=int(t)
        betas=self.betas.to(self.device)
        alphas_t= self.alphas.to(self.device)[t]
        sqrt_1_minus_alpha_bars = self.sqrt_1_minus_alpha_bars.to(self.device)[t]
        
        # noise
        z = torch.randn_like(xt) if t > 1 else 0
        noise = betas[t].sqrt()*z
        
        denoised_x = (xt - ((1 - alphas_t) * noise_pred)/sqrt_1_minus_alpha_bars)
        denoised_x= denoised_x/ (torch.sqrt(alphas_t))
        
        return denoised_x + noise

# note: sampling algorithm, Algorithm 2 Conditional sampling with classifier-free guidance
def sample(config, condition, num_intermediate_product=1):
    device=config['device']
    pretrained_model=config['pretrained_model']
    num_timesteps=int(config['num_timesteps'])
    num_input=condition.shape[0]
    guide_w=4 # guidence, # note: test on 0.2, 2, 4(best),  
    
    # setting:
    num_intermediate_product=1 if(num_intermediate_product>num_intermediate_product or num_intermediate_product==0) else num_intermediate_product # number of images to save in the denosie process
    condition_one_hot=torch.nn.functional.one_hot(condition, num_classes=20).to(torch.float32).to(device) # transform input lables to one hot map
    img_list=None # to save the output image during denoising process, a list of tensor[b,c,h,w]
        
    # load model
    model=torch.load(pretrained_model).to(device)
    model.eval()

    # random noise samples
    xt = torch.randn(condition.shape[0], 3, 28, 28).to(torch.float32).to(device)
    if num_intermediate_product!=1: 
        temp = torch.clamp(xt, -1., 1.)
        temp = (temp +1) / 2 
        img_list=temp.unsqueeze(0)

    # step: double input batch for guidence
    # print(condition_one_hot.shape)
    mask_out=condition_one_hot.clone()
    mask_out[:]=0
    condition_one_hot=torch.cat([condition_one_hot,mask_out],0).to(device)
    # print(condition_one_hot.shape)

    # step: denoise
    reverse_pass=DDPM_reverse_pass(T=num_timesteps)
    with torch.no_grad():
        for t in tqdm(reversed(range(num_timesteps))):
            
            # step: double input batch for guidence
            # print(xt.shape)
            xt = xt.repeat(2, 1, 1, 1)
            # print(xt.shape)
            # t=torch.tensor([t/num_timesteps], device=device)[:, None, None, None]
            # print(t.shape)
            time_i = torch.tensor([t/num_timesteps])[:, None, None, None].repeat(2*num_input, 1, 1, 1).to(device)
            # print(time_i.shape)
            
            # note: Algo2-3, predict noise
            noise_pred = model(xt, t=time_i, c=condition_one_hot)
            
            # split predictions and compute weighting
            eps1 = noise_pred[:num_input]  # condition
            eps2 = noise_pred[num_input:]  # uncondition
            noise_pred = (1 + guide_w) * eps1 - guide_w * eps2
            xt = xt[:num_input]
            
            # note: Algo2-4, 2-5, denoise
            xt = reverse_pass.sample_timestep_prev(xt=xt, noise_pred=noise_pred, t=t)
            
            # note: save img and convert img format
            if t% int(num_timesteps/num_intermediate_product)==0 :
                # print(f"saving intermediate product at step {t}")
                # temp= 2*xt - 1
                temp = torch.clamp(xt, -1., 1.)
                temp = (temp +1) / 2
                img_list=temp.unsqueeze(0) if img_list==None else torch.cat((img_list,temp.unsqueeze(0)),0)
            
    return img_list

# note: training with time and condition embedding, Algorithm 1 Joint training a diffusion model with classifier-free guidance
def train(config):
    print(config)
    mnistm_label=config['mnistm_label']    
    mnistm_imgs=config['mnistm_imgs']
    svhn_label=config['svhn_label']
    svhn_imgs=config['svhn_imgs']
    device=config['device']
    
    p_drop=float(config['p_drop'])
    epoch=int(config['epoch'])
    batch_size=int(config['batch_size'])
    lr=float(config['lr'])
    num_timesteps=int(config['num_timesteps'])
    gamma=float(config['gamma'])

    model_save_dir=config['model_save_dir']
    model_save_name=config['model_save_name']
    # print(f'Device: {device}\n')

    # step: Dataset
    mnistm_dataset=MNISTN_Dataset(annotations_file=mnistm_label,img_dir=mnistm_imgs, transform=TRANSFORM_MNISTM)
    svhn_dataset=SVHN_Dataset(annotations_file=svhn_label,img_dir=svhn_imgs, transform=TRANSFORM_SVHN)

    train_dataset=torch.utils.data.ConcatDataset([mnistm_dataset, svhn_dataset])
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size, shuffle=True, drop_last=True)
    SVHN_dataloader=DataLoader(svhn_dataset, batch_size=batch_size*2, shuffle=True, drop_last=False) # SVHM result alway worse then MNISTM
    MNISTM_dataloader=DataLoader(mnistm_dataset, batch_size=batch_size*2, shuffle=True, drop_last=False) # SVHM result alway worse then MNISTM
    
    # note:　img shape=([3, 28, 28])
    print(f"{(mnistm_dataset[0][0].shape)=}")
    print(f"{(svhn_dataset[0][0].shape)=}")
    print(f"{len(mnistm_dataset)=}")
    print(f"{len(svhn_dataset)=}")
    print(f"{len(train_dataset)=}")
    print(f"{len(train_dataloader)=}")
    
    # step: model_train
    model = ContextUnet(in_channels=3, height=28, width=28, n_feat=64, n_cfeat=20).to(device)

    # Diffusion Forward Process: add noise
    forward_pass = DDPM_forward_pass(T=num_timesteps)

    # Initialize Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Best Loss
    mean_loss_list=[]
    best_loss = float('inf')
    for ep in range(epoch):
        loss_list=[]
        model.train()

        # note: train on both dataset
        for data, labels in tqdm(train_dataloader):
            data=data.to(device)
            n_condition=labels.shape[0]
            n_masked_condition=int(n_condition*p_drop)
            mask=torch.ones(n_condition, 1).to(device)
            mask[:n_masked_condition]=0

            
            labels_one_hot=torch.nn.functional.one_hot(labels, num_classes=20).to(torch.float32).to(device)
            labels_one_hot=(labels_one_hot*mask)
            
            # Generate noise and timestamps
            noise = torch.randn_like(data).to(device)
            t = torch.randint(1, num_timesteps + 1, (data.shape[0],)).to(device) # should be 1~999

            # Add noise 
            noisy_imgs = forward_pass.add_noise(data, noise, t)
             
            # Predict noise 
            noise_pred = model(noisy_imgs, t=t/num_timesteps, c=labels_one_hot)
            print(noise_pred.shape)
            print(noise_pred[ :n_masked_condition].shape, noise[ :n_masked_condition].shape)
            # Calculate Loss
            loss_uncondition =  criterion(noise_pred[ :n_masked_condition], noise[ :n_masked_condition])
            loss_condition =    criterion(noise_pred[n_masked_condition: ], noise[n_masked_condition: ])
            loss= (1-gamma)* loss_uncondition + gamma * loss_condition
            loss_list.append(loss.item())

            # Backprop 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Mean Loss
        mean_epoch_loss = np.mean(loss_list)
        mean_loss_list.append(mean_epoch_loss)
        print(f"===epoch {ep}/{epoch}===\nmean train loss: {mean_epoch_loss}")
        
        # step: save model
        if mean_epoch_loss < best_loss:
            best_loss = mean_epoch_loss

            # delete worst model when buffer overflow
            if len(os.listdir(model_save_dir))>=MAX_SAVE_BUFFER: 
                loss, worst_model=find_model(model_save_dir,'loss', 'worst')
                os.remove(worst_model)
                print("Buffer overflow, removing worst model: "+ worst_model)
            
            # save model
            save_path=os.path.join(model_save_dir,f"ep{ep:03d}_{mean_epoch_loss}_{model_save_name}")
            torch.save(model, save_path)
            print(f"Saving: {model_save_dir=}\n{model_save_name=}\n{save_path=}")
            
    return mean_loss_list

if __name__=='__main__':
    # Test: DDPM_forward_pass
    # transform=transforms.ToTensor()
    # original = Image.open("hw2_data/digits/mnistm/data/00000.png")
    # original .save("no_noise.jpg")
    # original = transform(original).to('cuda')
    # original = original[None,:,:,:]
    # print(original.shape)
    # noise = torch.randn(1, 1, 28, 28).to('cuda')
    # t_steps = torch.randint(0, 10, (1,)).to('cuda')
    # dfp = DDPM_forward_pass()
    # out = dfp.add_noise(original, noise, t_steps)
    # out=transforms.functional.to_pil_image(out[0])
    # out.save("noise.jpg")

    # Test: DDPM_reverse_pass
    # original = torch.randn(1, 1, 28, 28).to('cuda')
    # noise_pred = torch.randn(1, 1, 28, 28).to('cuda')
    # t = torch.randint(0, 1000, (1,)).to('cuda') 
    # print(f"{t=}")
    # drp = DDPM_reverse_pass()
    # (out, x0) = drp.sample_prev_timestep(original, noise_pred, t)
    # print(out.shape)

    # TEST: find_worst_model()
    # dir_path="model_ckpt/p1/round2"
    # print(find_model(dir_path,'loss', 'best'))
    # print(find_model(dir_path,'loss', 'worst'))
    pass