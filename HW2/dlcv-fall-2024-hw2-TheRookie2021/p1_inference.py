import os
import torch
import matplotlib.pyplot as plt
import yaml
import argparse
# from p1_utils import MNISTN_Dataset, SVHN_Dataset, DDPM_forward_pass, DDPM_reverse_pass
from p1_utils import  sample, torch_to_PIL
from tqdm import tqdm
#=================================config=============================================#
config = yaml.safe_load(open("p1_config.yaml"))
# setup random seeds
torch.manual_seed(4202)
#=================================config_end=============================================#

#=================================arg_parse=============================================#
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='HW2 deep learning network.')
    parser.add_argument(
        '--number', type=int, default=50, help='num_save_for_each_condition')
    parser.add_argument(
        '--folder', type=str, default='Output_folder/', help='')
    parser.add_argument(
        '--model', default='', type=str, help='')
    parser.add_argument(
        '--note', type=str, help='')
    return parser.parse_args()
#=================================arg_parse_end=============================================#

if __name__ =='__main__':
    args=parse_args()
    config['p1_output_save_dir']=args.folder

    # step: prepare directory/folders for saving
    if not os.path.isdir(config["model_save_dir"]):
        os.makedirs(config["model_save_dir"])

    if not os.path.isdir(config["p1_output_save_dir"]):
        os.makedirs(config["p1_output_save_dir"])
    num_timesteps=int(config["num_timesteps"])
    # step: select and load model
    # config["pretrained_model"]=os.path.join(config["model_save_dir"], sorted(os.listdir(config["model_save_dir"]))[-1])
    config["pretrained_model"]=args.model
    model_id=config["pretrained_model"].split("_")[-1].split('.')[0]
    print("using model for generating samples: ", config["pretrained_model"])
    
    # step: generate conditions
    target=torch.arange(0,20)
    print(f"input conditions:, {target.shape=}, {target=}")
    
    # # step: for observation
    # num_intermediate_product=10
    # print(f"input conditions:, {target.shape=}, {target=}")
    # img_list=sample(config, target, num_intermediate_product=num_intermediate_product)
    
    # rows=10
    # for dataset_id in range(2):
    #     fig, axes = plt.subplots(nrows=rows, ncols=num_intermediate_product+1, figsize=(28, 28))
    #     for col, samples in tqdm(enumerate(img_list)): # 0~9, 10~19
    #         for row, s in zip(range(rows), samples[dataset_id*10:dataset_id*10+10] ):
    #             axes[row][col].imshow(torch_to_PIL(s, dataset_id)) 
    #             axes[row][col].axis('off')  # Turn off axis labels
    #     plt.subplots_adjust(wspace=0, hspace=0)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(config['p1_output_save_dir'],f"p1_obeservation_{model_id}_{num_timesteps}_{dataset_id}.png"))
    
    # step: p1-0, Sample random noise from normal distribution to generate 50 conditional images for each digit (0-9) on MNIST-M & SVHN datasetsplot 10 imgs for each conditions
    conditions=None
    num_save_for_each_condition=int(args.number)
    n_total_imgs= num_save_for_each_condition*20
    batch_size=80 # sample 10*20 for each time (avoid put all 1000 imgs into gpu at once )
    iteration=int(n_total_imgs/batch_size)
    n_replicate=int(batch_size/20)
    print('preparing P1_0')
    for i in range(n_replicate):
        conditions=target if conditions==None else torch.cat((conditions,target),0)
    print(f"{n_total_imgs=}")
    print(f"{batch_size=}")
    print(f"{iteration=}")
    print(f"{conditions.shape=}")
    print(f"{n_replicate=}")
    
    for it in tqdm(range(iteration)):
        P1_0_imgs=sample(config, conditions, num_intermediate_product=0)
        print(P1_0_imgs.shape)
        P1_0_imgs=torch.squeeze(P1_0_imgs)
        print(P1_0_imgs.shape)
        P1_0_imgs=torch.reshape(P1_0_imgs, (n_replicate, 20, P1_0_imgs.shape[1],P1_0_imgs.shape[2],P1_0_imgs.shape[3]))
        print(P1_0_imgs.shape)
        # saving: outputfolder/dataset_id/condition_id/img.png
        for i, twenty_conditions in enumerate(P1_0_imgs): # condition 1~20
            for condition_id, s in enumerate(twenty_conditions):
                dataset_id=int(condition_id/10)
                save_dir=os.path.join(config['p1_output_save_dir'],'mnistm')if dataset_id==0 else os.path.join(config['p1_output_save_dir'],'svhn')
                if not os.path.isdir(save_dir): os.makedirs(save_dir)
                # print(save_dir)
                torch_to_PIL(s, int(condition_id/10)).save(os.path.join(save_dir,f'{condition_id%10}_{(it*n_replicate+i):03d}.png'))

    # step: p1-2 10 for each condition, two dataset plt fig
    # num_save_for_each_condition=10
    # P1_2_img_list=[]
    # print('preparing P1_2')
    # for i in tqdm(range(num_save_for_each_condition)):
    #     P1_2_img_list.append(sample(config, target, num_intermediate_product=0))
    # rows=10
    # fig0, axes0 = plt.subplots(nrows=rows, ncols=10, figsize=(28, 28))
    # fig1, axes1 = plt.subplots(nrows=rows, ncols=10, figsize=(28, 28))
    # for i, sample_conditions in enumerate(P1_2_img_list): # condition 1~20
    #     for intermediate in sample_conditions:
    #         for condition_id, s in enumerate(intermediate):
    #             dataset_id=int(condition_id/10)
    #             if dataset_id==0:
    #                 axes0[condition_id][i].imshow(torch_to_PIL(s )) 
    #                 axes0[condition_id][i].axis('off')  # Turn off axis labels
    #             else:
    #                 axes1[condition_id%10][i].imshow(torch_to_PIL(s )) 
    #                 axes1[condition_id%10][i].axis('off')  # Turn off axis labels
    # fig0.subplots_adjust(wspace=0, hspace=0)
    # fig0.tight_layout()
    # fig0.savefig(os.path.join(config['p1_output_save_dir'],f"p1_2_intermediate_0.png"))
    # fig1.subplots_adjust(wspace=0, hspace=0)
    # fig1.tight_layout()
    # fig1.savefig(os.path.join(config['p1_output_save_dir'],f"p1_2_intermediate_1.png"))

    # step: p1-3 condition 0, 10 intermediate

    # num_save_for_each_condition=10 # note: 10*20 imgs
    # conditions=None
    # print('preparing P1_3')
    # for i in range(num_save_for_each_condition):
    #     conditions=target if conditions==None else torch.cat((conditions,target),0)
    # P1_3_img_list=sample(config, conditions, num_intermediate_product=0)
    

    # print(P1_3_img_list.shape)
    # P1_3_img_list=torch.squeeze(P1_3_img_list)
    # print(P1_3_img_list.shape)
    # P1_3_img_list=torch.reshape(P1_3_img_list, (10, 20,P1_3_img_list.shape[1],P1_3_img_list.shape[2],P1_3_img_list.shape[3]))
    # rows=10
    # fig0, axes0 = plt.subplots(nrows=rows, ncols=num_save_for_each_condition, figsize=(28, 28))
    # fig1, axes1 = plt.subplots(nrows=rows, ncols=num_save_for_each_condition, figsize=(28, 28))
        
    # for col, intermediates in enumerate(P1_3_img_list): # condition 1~20
    #     for condition_id, s in enumerate(intermediates):
    #         dataset_id=int(condition_id/10)
    #         if dataset_id==0:
    #             axes0[condition_id][col].imshow(torch_to_PIL(s )) 
    #             axes0[condition_id][col].axis('off')  # Turn off axis labels
    #         else:
    #             axes1[condition_id%10][col].imshow(torch_to_PIL(s )) 
    #             axes1[condition_id%10][col].axis('off')  # Turn off axis labels

    # fig0.subplots_adjust(wspace=0, hspace=0)
    # fig0.tight_layout()
    # fig0.savefig(os.path.join(config['p1_output_save_dir'],f"p1_3_all_{model_id}_0.png"))
    # fig1.subplots_adjust(wspace=0, hspace=0)
    # fig1.tight_layout()
    # fig1.savefig(os.path.join(config['p1_output_save_dir'],f"p1_3_all_{model_id}_1.png"))
    
            
    # # step: p1-3, Visualize a total of six images from both MNIST-M & SVHN datasets in the reverse process of the first “0” in your outputs in (2) and with different time steps
    # print('preparing P1_3')
    # rows=1
    # num_intermediate_product=5
    # timesteps_list=[0,200,400,600,800,1000]
    # P1_3_img_list=sample(config, target, num_intermediate_product=num_intermediate_product)

    # for dataset_id in range(2):
    #     fig, axes = plt.subplots(nrows=rows, ncols=num_intermediate_product+1, )
    #     for col, samples in tqdm(enumerate(P1_3_img_list)): 
    #         if dataset_id==0:
    #             axes[col].imshow(torch_to_PIL(samples[0] )) 
    #             axes[col].set_title(f"t={timesteps_list[col]}")
    #             axes[col].axis('off')  # Turn off axis labels
    #             plt.subplots_adjust(wspace=0, hspace=0)
    #             plt.tight_layout()
    #             plt.savefig(os.path.join(config['p1_output_save_dir'],f"p1_3_0_{num_timesteps}_{dataset_id}.png"))
    #         else:
    #             axes[col].imshow(torch_to_PIL(samples[10] )) 
    #             axes[col].set_title(f"t={timesteps_list[col]}")
    #             axes[col].axis('off')  # Turn off axis labels
    #             plt.subplots_adjust(wspace=0, hspace=0)
    #             plt.tight_layout()
    #             plt.savefig(os.path.join(config['p1_output_save_dir'],f"p1_3_0_{num_timesteps}_{dataset_id}.png"))
                

    