import argparse, os, sys, glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from torchvision.utils import make_grid
from omegaconf import OmegaConf
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append('stable-diffusion/ldm/models/diffusion')
# from ddpm import DiffusionWrapper
# from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.util import instantiate_from_config
from p1_utils import DDPM_forward_pass
from p3_utils import TextualInversionDataset
TRANSFORM= transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5], [0.5,0.5])
            
        ]) 

# ============================================================================================================================================
# step: config
DEBUG_FLAG=True

ckpt="HW2/P2_3_model.pth"
device='cuda' if torch.cuda.is_available() else 'cpu'
if DEBUG_FLAG: print('========DEBUG========')

# encode embeddings
# step: load model, ldm.models.diffusion.ddpm.LatentDiffusion
LDM=torch.load(ckpt).to(device) 
torch.save(LDM.cond_stage_model.transformer, os.path.join("model_ckpt/p3","personalized_embeding"))