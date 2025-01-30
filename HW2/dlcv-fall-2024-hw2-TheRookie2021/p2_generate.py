import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse

from UNet import UNet

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='HW2 deep learning network.')
    parser.add_argument(
        '--input_folder', type=str, default='hw2_data/face/noise', help='')
    parser.add_argument(
        '--output_folder', type=str, default='p2_outputs_evaluation', help='')
    parser.add_argument(
        '--model_path', default='"hw2_data/face/UNet.pt"', type=str, help='')
    parser.add_argument(
        '--note', type=str, help='')
    return parser.parse_args()

def beta_scheduler(Timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start, linear_end, Timestep, dtype=torch.float64)
    return betas
def slerp(x0, x1, alpha):
    theta=torch.arccos( torch.matmul(x0,x1) / (torch.norm(x0)*torch.norm(x1)))
    return (torch.sin( (1-alpha)*theta )*x0/ torch.sin(theta)) + (torch.sin( alpha*theta )*x1/ torch.sin(theta))  
def linear(x0, x1, alpha):
    return torch.lerp(x0, x1, alpha)

# ========================  DDIM   ========================
class DDIM:
    def __init__(self, model, timesteps=1000, beta_schedule=beta_scheduler()):
        self.model = model
        self.timesteps = timesteps
        self.betas = beta_schedule
        self.alphas = 1.0 - self.betas
        self.alphas_bars = torch.cumprod(self.alphas, axis=0)
        self.alphas_bars_prev = F.pad(self.alphas_bars[:-1], (1, 0), value=1.0)

        # step: q(x_t | x_{t-1})
        self.sqrt_alphas_bars = torch.sqrt(self.alphas_bars)
        self.sqrt_recip_alphas_bars = torch.sqrt(1.0 / self.alphas_bars)
        self.sqrt_recipm1_alphas_bars = torch.sqrt(1.0 / self.alphas_bars - 1)
        self.sqrt_one_minus_alphas_bars = torch.sqrt(1.0 - self.alphas_bars)
        self.log_one_minus_alphas_bars = torch.log(1.0 - self.alphas_bars)

        # step: posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_bars_prev) / (1.0 - self.alphas_bars))
        
        # step: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_bars_prev)/ (1.0 - self.alphas_bars))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_bars_prev) * torch.sqrt(self.alphas)/ (1.0 - self.alphas_bars))
    
                    
    def interploate_two_noise(self, config, alpha ,id_1, id_2, operation=slerp):
        if id_1 == None or id_2== None: return None
        batch_size= 1
        ddim_timesteps= int(config["ddim_timesteps"])
        eta= float(config["eta"])
        print(f"{eta=}")
        input_folder= config["input_folder"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        c = int( self.timesteps /ddim_timesteps)
        
        # add one to get the final alpha values during sampling
        ddim_timestep_seq = np.array([i for i in range(0, self.timesteps, c)])
        ddim_timestep_seq = ddim_timestep_seq + 1

        # get previous
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        filenames = sorted(os.listdir(input_folder))
        input_pt = [torch.load(os.path.join(input_folder, filename)) for filename in filenames]
        # print(input_pt[0].shape)

        # note: interpolation
        input_pt = operation(input_pt[id_1].to(device),input_pt[id_2].to(device), alpha=alpha.to(device))

        with torch.no_grad():
            for i in tqdm(reversed(range(0, ddim_timesteps)), total=ddim_timesteps,):
                t = torch.full((batch_size,), ddim_timestep_seq[i]).to(torch.long).to(device)
                prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i]).to(torch.long).to(device)

                # note: 1. current and previous alpha_bars
                alpha_bars_t = self.alphas_bars.to(device).gather(0, t).float().reshape(batch_size, *((1,) * (len(input_pt.shape) - 1)))
                alpha_bars_t_prev = self.alphas_bars.to(device).gather(0, prev_t).float().reshape(batch_size, *((1,) * (len(input_pt.shape) - 1)))
        
                # note: 2. predict noise 
                pred_noise = self.model(input_pt, t)

                # note: 3. get the predicted x_0
                pred_x0 = (input_pt - torch.sqrt((1.0 - alpha_bars_t)) * pred_noise) / torch.sqrt(alpha_bars_t)
                pred_x0 = torch.clamp(pred_x0, -1.,1.)

                # note: 4. compute variance: "sigma_t(η)" -> see formula (16), σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
                sigmas_t = eta * torch.sqrt((1 - alpha_bars_t_prev)/ (1 - alpha_bars_t)* (1 - alpha_bars_t / alpha_bars_t_prev))

                # note: 5. compute "direction pointing to x_t" of formula (12)
                pred_dir_xt = (torch.sqrt(1 - alpha_bars_t_prev - sigmas_t**2) * pred_noise)

                # note: 6. compute x_{t-1} of formula (12)
                x_prev = (torch.sqrt(alpha_bars_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(input_pt))
                input_pt = x_prev

        return filenames, input_pt
            
    def sample(self, config):
        batch_size= int(config["batch_size"])
        ddim_timesteps= int(config["ddim_timesteps"])
        eta= float(config["eta"])
        # print(f"{eta=}")
        input_folder= config["input_folder"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        c = int( self.timesteps /ddim_timesteps)
        
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        # ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        ddim_timestep_seq = np.array([i for i in range(0, self.timesteps, c)])
        ddim_timestep_seq = ddim_timestep_seq + 1

        # get previous
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        # start from pure noise (for each example in the batch)
        filenames = sorted(os.listdir(input_folder))
        # read all pt file
        input_pt = [torch.load(os.path.join(input_folder, filename)) for filename in filenames]
        # print(input_pt[0].shape)

        input_pt = torch.cat(input_pt, dim=0)
        # print(input_pt.shape)

        with torch.no_grad():
            for i in tqdm(reversed(range(0, ddim_timesteps)), total=ddim_timesteps,):
                t = torch.full((batch_size,), ddim_timestep_seq[i]).to(torch.long).to(device)
                prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i]).to(torch.long).to(device)

                # note: 1. current and previous alpha_bars
                alpha_bars_t = self.alphas_bars.to(device).gather(0, t).float().reshape(batch_size, *((1,) * (len(input_pt.shape) - 1)))
                alpha_bars_t_prev = self.alphas_bars.to(device).gather(0, prev_t).float().reshape(batch_size, *((1,) * (len(input_pt.shape) - 1)))
        
                # note: 2. predict noise 
                pred_noise = self.model(input_pt, t)

                # note: 3. get the predicted x_0
                pred_x0 = (input_pt - torch.sqrt((1.0 - alpha_bars_t)) * pred_noise) / torch.sqrt(alpha_bars_t)
                pred_x0 = torch.clamp(pred_x0, -1.,1.)

                # note: 4. compute variance: "sigma_t(η)" -> see formula (16), σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
                sigmas_t = eta * torch.sqrt((1 - alpha_bars_t_prev)/ (1 - alpha_bars_t)* (1 - alpha_bars_t / alpha_bars_t_prev))

                # note: 5. compute "direction pointing to x_t" of formula (12)
                pred_dir_xt = (torch.sqrt(1 - alpha_bars_t_prev - sigmas_t**2) * pred_noise)

                # note: 6. compute x_{t-1} of formula (12)
                x_prev = (torch.sqrt(alpha_bars_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(input_pt))
                input_pt = x_prev

        return filenames, input_pt

class Face_Dataset(Dataset):
    def __init__(self, input_dir, gt_dir, transform=None, target_transform=None):
        self.input_dir =input_dir
        self.gt_dir= gt_dir
        
        self.input_imgs = [i for i in sorted(os.listdir(input_dir)) if i.endswith('png') or i.endswith('jpg')]
        self.gt_imgs= sorted(os.listdir(gt_dir))
        self.transform = transform
        # print(self.input_imgs)
        # print(self.gt_imgs)
    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_imgs[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_imgs[idx])
        input_img = Image.open(input_path)
        gt_img = Image.open(gt_path)
        
        if self.transform!=None:
            input_img = self.transform(input_img)
            gt_img = self.transform(gt_img)
        print(input_img.shape, gt_img.shape)
        return input_img, gt_img
    
    def __len__(self):
        return len(self.input_imgs)

def generate(config):
    # step: config
    T = int(config["time"])
    model_path = config["model_path"]
    output_folder = config["output_folder"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir(output_folder): os.makedirs(output_folder)
    
    # step: load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path))

    ddim = DDIM( model=model, timesteps=T)

    with torch.no_grad():
        # step: sample
        x_names, x_gen = ddim.sample(config) 
        # step: save img
        for i, img in enumerate(x_gen):
            img= torch.clamp(img, min=-1,max = 1) # Normalization
            img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
            torchvision.utils.save_image(img, os.path.join(output_folder, f"{i:02d}.png"))
            

def generate_interpolate(config, id_1=None, id_2=None, operation=slerp):
    # step: config
    T = int(config["time"])
    model_path = config["model_path"]
    output_folder = config["output_folder"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alphas=torch.arange(0,1.1,0.1)
    print(alphas)
    if not os.path.isdir(output_folder): os.makedirs(output_folder)
    
    # step: load model
    model = UNet()
    model.load_state_dict(torch.load(model_path))

    ddim = DDIM( model=model.to(device), timesteps=T)

    with torch.no_grad():
        # step: sample
        for a in alphas:
            x_names, x_gen = ddim.interploate_two_noise( config, alpha=a, id_1=id_1, id_2=id_2, operation=operation)

            # step: save img
            img = x_gen
            img= torch.clamp(img, min=-1,max = 1) # Normalization
            img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
            torchvision.utils.save_image(img, os.path.join(output_folder, f"alpha{a:2f}.png"))

def evaluate(input_dir, gt_dir):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    criterion = torch.nn.MSELoss()
    TRANSFORM= transforms.Compose([
            transforms.ToTensor(),
        ]) 
    dataset=Face_Dataset(input_dir, gt_dir, transform=TRANSFORM)
    datalader=DataLoader(dataset, batch_size=10, shuffle=None)
    loss_list=[]
    for inputs, gts in datalader:
        inputs, gts= inputs.to(device), gts.to(device)
        loss = criterion(inputs, gts)
        loss_list.append(loss.item())
    mean_epoch_loss = np.mean(loss_list)
    print(f"mean train loss: {mean_epoch_loss}")
    return mean_epoch_loss

if __name__ == "__main__":
    args=parse_args()
    config = yaml.safe_load(open("p2_config.yaml"))
    config["input_folder"]=args.input_folder
    config["output_folder"]=args.output_folder
    config["model_path"]=args.model_path
    
    # step: evaluate on eta=0
    generate(config)
    # print('evaluate on eta=0')
    # evaluate(input_dir=config["output_folder"], gt_dir="hw2_data/face/GT")


    # step: generate all eta
    # # eta 5 steps
    # eta_step=[i/int(config["eta_step"]) for i in range(int(config["eta_step"])+1)]
    # print(eta_step)
    # for eta in eta_step:
    #     config["eta"]=eta
    #     config["output_folder"]=os.path.join("p2", str(int(eta*100)))
    #     generate(config)

    # step: evaluate on all eta
    # print('evaluate on all eta=0')
    # folders=os.listdir(config["output_folder"])
    # for subdir in folders:
    #     print(subdir)
    #     evaluate(input_dir=os.path.join(config["output_folder"],subdir), gt_dir="hw2_data/face/GT")


    
    # # step: plot 2.1 Please generate face images of noise 00.pt ~ 03.pt with different eta in one grid. 
    # output_folder = config["output_folder"]
    # sub_dir=[os.path.join(output_folder, s) for s in sorted(os.listdir('p2'))]
    
    # print(sub_dir)
    # num_img=4
    # fig, axes = plt.subplots(nrows=len(sub_dir), ncols=num_img, figsize=(num_img,len(sub_dir)) )
    # for row, dir in enumerate(sub_dir): # folder name: eta
    #     img_path=[os.path.join(dir, img) for img in sorted(os.listdir(dir))]
    #     print(img_path)
    #     axes[row][0].set_title(f"eta={dir[3:]}",  loc='left') 
    #     for col, img in enumerate(img_path[:num_img]): # 00~03 files
    #         face = mpimg.imread(img)
    #         axes[row][col].imshow(face)
    #         axes[row][col].axis('off')
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    # plt.tight_layout(pad=1)
    # plt.savefig('p2_1.png')

    # # # step: plot 2.2 Please generate the face images of the interpolation of noise 00.pt ~ 01.pt.        
    # config["eta"]=0
    # config["output_folder"]="p2_2/slerp"
    # generate_interpolate(config,  id_1=0, id_2=1, operation=slerp)
    # files=[os.path.join(config["output_folder"], img) for img in sorted(os.listdir(config["output_folder"]))]
    # fig, axes = plt.subplots(nrows=1, ncols=len(files),figsize=(len(files),1))
    # for i,img_path in enumerate(files):
    #     face = mpimg.imread(img_path)
    #     axes[i].imshow(face)
    #     axes[i].axis('off')
    #     axes[i].set_title(f"a=0.{img_path.split('.')[1][:2]}")
    # plt.subplots_adjust(0)
    # plt.tight_layout(pad=0)
    # plt.savefig('p2_2_slerp.png')


    # config["output_folder"]="p2_2/linear"
    # generate_interpolate(config,  id_1=0, id_2=1, operation=linear)
    # files=[os.path.join(config["output_folder"], img) for img in sorted(os.listdir(config["output_folder"]))]
    # fig, axes = plt.subplots(nrows=1, ncols=len(files),figsize=(len(files),1))
    # for i,img_path in enumerate(files):
    #     face = mpimg.imread(img_path)
    #     axes[i].imshow(face)
    #     axes[i].axis('off')
    #     axes[i].set_title(f"a=0.{img_path.split('.')[1][:2]}")
    # plt.subplots_adjust(0)
    # plt.tight_layout(pad=0)
    # plt.savefig('p2_2_linear.png')
