# from tqdm import tqdm, trange
# from torchvision.utils import make_grid
import argparse, os, sys
import torch
import numpy as np
from PIL import Image
import json
from einops import rearrange
from tqdm import tqdm

sys.path.append('stable-diffusion/ldm/models/diffusion')
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='HW2 deep learning network.')
    parser.add_argument(
        '--number', type=int, default=25, help='num_save_for_each_condition')
    parser.add_argument(
        '--input_json', type=str, default='hw2_data/textual_inversion/input.json', help='')
    parser.add_argument(
        '--output_folder', type=str, default='p3_outputs_evaluation', help='')
    parser.add_argument(
        '--model', default='HW2/P2_3_model.pth', type=str, help='')
    parser.add_argument(
        '--resume', type=str,default=None, help='')
    return parser.parse_args()
#=================================arg_parse_end=============================================#

if __name__ =='__main__':
    torch.manual_seed(58)
    args=parse_args()

    ckpt="HW2/P2_3_model.pth"
    device='cuda' if torch.cuda.is_available() else 'cpu'

    # step: dealing with prompt, 1. parsing .json 2. format the prompts to list
    # PROMPT_0=["A <new1> shepherd posing proudly on a hilltop with Mount Fuji in the background.", "A <new1> perched on a park bench with the Colosseum looming behind."]
    # PROMPT_1=["The streets of Paris in the style of <new2>.", "Manhattan skyline in the style of <new2>."]
    with open(args.input_json, 'r') as file:
        data = json.load(file)
    PROMPT_0=data[list(data.keys())[0]]['prompt']
    PROMPT_1=data[list(data.keys())[1]]['prompt']
    tokens_list=list(data.keys())
    output_folder=args.output_folder
    output_folder_token=[[],[]]
    for i, _ in enumerate(PROMPT_0):
        output_folder_token[0].append( os.path.join(  os.path.join(output_folder,str(0)), str(i) )  )
    for i, _ in enumerate(PROMPT_1):
        output_folder_token[1].append( os.path.join(  os.path.join(output_folder,str(1)), str(i) )  )

    print(output_folder_token)
    for i in output_folder_token:
        for j in i:
            os.makedirs(j, exist_ok=True)
            print(f"make dir: {j}")

    saving_counter_0=[0 for i in PROMPT_0] # use for setting filenames
    saving_counter_1=[0 for i in PROMPT_1]
    
    # step: load model, ldm.models.diffusion.ddpm.LatentDiffusion
    model=torch.load(ckpt).to(device) 

    # step: config for sampling
    total_img= 25 # note: 25 images for each prompt

    mini_batch= [4]*(total_img//4)+[total_img%4]
    print(mini_batch)
    shape = [4, 512 // 8, 512 // 8]
    # print(f"{mini_batch=}")
    sampler = DPMSolverSampler(model)

    # note: generate token 0
    for prompt_id, prompt in tqdm(enumerate(PROMPT_0)):
        # mini_batch= [4]*(total_img//4)+[total_img%4]
        counter=0
        # print(mini_batch)
        for index, batch_size in enumerate(mini_batch):
            input_prompts = batch_size * [prompt]
            # print(input_prompts)

            c = model.get_learned_conditioning(input_prompts)
            uc = model.get_learned_conditioning( len(input_prompts)*[""])
            
            print(f"sampling...")
            samples_ddim, _ = sampler.sample(   S=50,
                                                conditioning=c,
                                                batch_size=len(input_prompts),
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=7.5,
                                                unconditional_conditioning=uc,
                                                eta=0,
                                                x_T=None)

            # print(samples_ddim.shape)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            # print(x_samples_ddim.shape)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            # print(x_samples_ddim.shape)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
            # print(x_samples_ddim.shape)
            x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
            
            print(f"saving img...")
            for id, x_sample in enumerate(x_checked_image_torch):
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                save_name=f"source0_prompt{prompt_id}_{counter}.png"
                saving_des=os.path.join(output_folder_token[0][prompt_id], save_name)
                # print(saving_des)
                img.save(saving_des)
                counter+=1
    
    # note: generate token 1 
    for prompt_id, prompt in tqdm(enumerate(PROMPT_1)):
        # mini_batch= [4]*(total_img//4)+[total_img%4]
        counter=0
        # print(mini_batch)
        for index, batch_size in enumerate(mini_batch):
            input_prompts = batch_size * [prompt]
            # print(input_prompts)

            c = model.get_learned_conditioning(input_prompts)
            uc = model.get_learned_conditioning(len(input_prompts)*[""])
            
            print(f"sampling...")
            samples_ddim, _ = sampler.sample(   S=50,
                                                conditioning=c,
                                                batch_size=len(input_prompts),
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=8,
                                                unconditional_conditioning=uc,
                                                eta=0,
                                                x_T=None)

            # print(samples_ddim.shape)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            # print(x_samples_ddim.shape)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            # print(x_samples_ddim.shape)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
            print(x_samples_ddim.shape)
            x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
            
            print(f"saving img...")
            for id, x_sample in enumerate(x_checked_image_torch):
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(x_sample.astype(np.uint8))
                save_name=f"source0_prompt{prompt_id}_{counter}.png"
                saving_des=os.path.join(output_folder_token[1][prompt_id], save_name)
                # print(saving_des)
                img.save(saving_des)
                counter+=1
                        