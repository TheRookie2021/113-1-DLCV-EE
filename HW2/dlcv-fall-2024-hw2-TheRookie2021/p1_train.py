import os
import torch
import matplotlib.pyplot as plt
import yaml
from p1_utils import train
#=================================config=============================================#
config = yaml.safe_load(open("p1_config.yaml"))
# setup random seeds
torch.manual_seed(58)
# prepare directory/folders for saving
if not os.path.isdir(config["model_save_dir"]):
    os.makedirs(config["model_save_dir"])

if not os.path.isdir(config["p1_output_save_dir"]):
    os.makedirs(config["p1_output_save_dir"])

#=================================config_end=============================================#
if __name__ =='__main__':
    # step: train
    loss=train(config)
    plt.plot(range(len(loss)), loss)
    plt.savefig(os.path.join(config["p1_output_save_dir"],"loss_fig-2.png"))