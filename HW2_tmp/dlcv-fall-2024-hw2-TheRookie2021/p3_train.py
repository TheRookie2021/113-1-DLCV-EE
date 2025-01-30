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
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR, StepLR

sys.path.append('stable-diffusion/ldm/models/diffusion')
# from ddpm import DiffusionWrapper
# from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.util import instantiate_from_config
from p3_utils import TextualInversionDataset
TRANSFORM= transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5], [0.5,0.5])
            
        ]) 
    

def load_model_from_config(config, ckpt, verbose=False):
    if DEBUG_FLAG: print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        if DEBUG_FLAG: print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        if DEBUG_FLAG: print("missing keys:")
        if DEBUG_FLAG: print(m)
    if len(u) > 0 and verbose:
        if DEBUG_FLAG: print("unexpected keys:")
        if DEBUG_FLAG: print(u)

    model.cuda()
    model.eval()
    return model
# ============================================================================================================================================
# step: config
DEBUG_FLAG=False
resume="model_ckpt/p3/best_pass_3_outof_4.pth"

save_path="model_ckpt/p3"
if not os.path.isdir(save_path): os.makedirs(save_path)
dog_dir="hw2_data/textual_inversion/0"
David_Revoy_dir="hw2_data/textual_inversion/1"

batch_size=4
epoch=20
lr=1e-6
repeats=50
# weight_decay=0.0001
num_train_timesteps=1000
config = OmegaConf.load("stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
ckpt="stable-diffusion/ldm/models/stable-diffusion-v1/model.ckpt"
device='cuda' if torch.cuda.is_available() else 'cpu'
if DEBUG_FLAG: print('========DEBUG Mode========')

# encode embeddings
# step: load model, ldm.models.diffusion.ddpm.LatentDiffusion
if resume==None:
    print(f"loading model: {ckpt}")
    LDM=load_model_from_config(config, ckpt).to(device) 
else:
    print(f"reloading model: {resume}")
    LDM=torch.load(resume).to(device) 

if DEBUG_FLAG: print(f"initial embeddings{LDM.cond_stage_model.transformer.get_input_embeddings()}")

# step: freeze unet, transformer

if DEBUG_FLAG: print("LDM components:")
if DEBUG_FLAG: print(f"\t{LDM.__class__.__name__=}") # LatentDiffusion
if DEBUG_FLAG: print(f"\t{LDM.model.__class__.__name__=}") # DiffusionWrapper
if DEBUG_FLAG: print(f"\t{LDM.model.diffusion_model.__class__.__name__=}") # UNetModel
if DEBUG_FLAG: print(f"\t{LDM.first_stage_model.__class__.__name__=}") # AutoencoderKL
if DEBUG_FLAG: print(f"\t{LDM.cond_stage_model.__class__.__name__=}") # FrozenCLIPEmbedder
if DEBUG_FLAG: print(f"\t{LDM.cond_stage_model.tokenizer.__class__.__name__=}") # CLIPTokenizer
if DEBUG_FLAG: print(f"\t{LDM.cond_stage_model.transformer.__class__.__name__=}") # CLIPTextModel

# ? How to update embedding?

# step: preparing input
new_token=["<new1>", "<new2>"]
placeholder_tokens = new_token

# add dummy tokens for multi-vector
if resume==None:
    additional_tokens = []
    for i in range(len(placeholder_tokens) ):
        additional_tokens.append(f"{placeholder_tokens[i]}_{i}")
    placeholder_tokens += additional_tokens
    # add new token to tokenizer
    num_added_token=LDM.cond_stage_model.tokenizer.add_tokens(placeholder_tokens)

placeholder_token_ids = LDM.cond_stage_model.tokenizer.convert_tokens_to_ids(placeholder_tokens)
if DEBUG_FLAG: print(f"{placeholder_token_ids=}")

if resume==None:
    # resize the embedding for newly added tokens
    if DEBUG_FLAG: print(f"{len(LDM.cond_stage_model.tokenizer)=}")
    LDM.cond_stage_model.transformer.resize_token_embeddings(len(LDM.cond_stage_model.tokenizer))
    token_embeds = LDM.cond_stage_model.transformer.get_input_embeddings().weight.data

    # initialized token with related concept
    initializer_token_id = LDM.cond_stage_model.tokenizer.encode(["dog","drawing","",""], add_special_tokens=False)

    with torch.no_grad():
        for i, token_id in enumerate(placeholder_token_ids):
            # if DEBUG_FLAG: print(f"{token_id}: {initializer_token_id[i]}")
            token_embeds[token_id] = token_embeds[initializer_token_id[i]].clone()
            token_embeds[token_id].requires_grad=True
token_embeds = LDM.cond_stage_model.transformer.get_input_embeddings().weight.data
orig_embeds_params = token_embeds.clone()

if DEBUG_FLAG: print(f"before initialized weight: {token_embeds.shape=}")
if DEBUG_FLAG: print(f"before initialized weight: {token_embeds[min(placeholder_token_ids) : max(placeholder_token_ids)+1].shape=}")
if DEBUG_FLAG: print(f"before initialized weight: {token_embeds[min(placeholder_token_ids) : max(placeholder_token_ids)+1]=}")
if DEBUG_FLAG: print(f"after initialized weight: {token_embeds[min(placeholder_token_ids) : max(placeholder_token_ids)+1]=}")

if DEBUG_FLAG: print(f"{token_embeds.shape=}")
if DEBUG_FLAG: print(f"{placeholder_tokens=}")
if DEBUG_FLAG: print(f"{placeholder_token_ids=}")
if DEBUG_FLAG: print(f"{num_added_token=}")
if DEBUG_FLAG: print(f"{LDM.cond_stage_model.tokenizer.convert_ids_to_tokens(placeholder_token_ids[0])=}")
if DEBUG_FLAG: print(f"added embeddings{LDM.cond_stage_model.transformer.get_input_embeddings()}")

dog_dataset=TextualInversionDataset(
        data_root=dog_dir,
        tokenizer=LDM.cond_stage_model.tokenizer,
        # size=args.resolution,
        placeholder_token=(LDM.cond_stage_model.tokenizer.convert_ids_to_tokens(placeholder_token_ids[0])),
        repeats=repeats,
        learnable_property="object",
        center_crop=False,
        set="train",
    )
david_dateset=TextualInversionDataset(
        data_root=David_Revoy_dir,
        tokenizer=LDM.cond_stage_model.tokenizer,
        # size=args.resolution,
        placeholder_token=(LDM.cond_stage_model.tokenizer.convert_ids_to_tokens(placeholder_token_ids[1])),
        repeats=repeats,
        learnable_property="style",
        center_crop=False,
        set="train",
    )
# train_datasets=torch.utils.data.ConcatDataset([dog_dataset, david_dateset])
dog_train_dataloader=DataLoader(dog_dataset, batch_size=batch_size, shuffle=True)
david_train_dataloader=DataLoader(david_dateset, batch_size=batch_size, shuffle=True)

# step: train
weight_dtype = torch.float32
# optimizer= LDM.configure_optimizers()
# optimizer = torch.optim.Adam(LDM.cond_stage_model.transformer.text_model.embeddings.token_embedding.parameters(), lr=lr)
LDM.cond_stage_model.transformer.text_model.embeddings.token_embedding.train()
optimizer = torch.optim.AdamW(
        LDM.cond_stage_model.transformer.text_model.embeddings.token_embedding.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

# lambda1 = lambda epoch:0.95 ** epoch 
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)
scheduler = StepLR(optimizer, step_size=epoch//4, gamma=0.1)

criterion = torch.nn.MSELoss(reduction="mean")
# ====freeze====
# LDM.cond_stage_model.transformer.text_model.encoder.requires_grad=False
# LDM.cond_stage_model.transformer.text_model.final_layer_norm.requires_grad=False
# LDM.cond_stage_model.transformer.text_model.embeddings.position_embedding.requires_grad=False
# LDM.cond_stage_model.transformer.text_model.embeddings.token_embedding.requires_grad=True

# for param in LDM.model.diffusion_model.parameters():
#     param.requires_grad = False
# for param in LDM.first_stage_model.parameters():
#     param.requires_grad = False
# LDM.model.diffusion_model.eval()
# LDM.first_stage_model.eval()

LDM.model.diffusion_model.requires_grad=False
LDM.first_stage_model.requires_grad=False
LDM.cond_stage_model.transformer.text_model.encoder.requires_grad_(False)
LDM.cond_stage_model.transformer.text_model.final_layer_norm.requires_grad_(False)
LDM.cond_stage_model.transformer.text_model.embeddings.position_embedding.requires_grad_(False)
# LDM.cond_stage_model.transformer.text_model.embeddings.token_embedding.requires_grad_(True)
# LDM.cond_stage_model.transformer.get_input_embeddings().requires_grad_(True)



# if DEBUG_FLAG: print(f"{LDM.cond_stage_model.transformer.text_model.embeddings.token_embedding.requires_grad=}")
if DEBUG_FLAG: print(orig_embeds_params.shape)
index_no_updates = torch.ones((len(LDM.cond_stage_model.tokenizer),), dtype=torch.bool)
index_no_updates[min(placeholder_token_ids) :  max(placeholder_token_ids) + 1] = False


# LDM.cond_stage_model.transformer.text_model.embeddings.token_embedding.train()
for param in LDM.cond_stage_model.transformer.parameters():
    print(param.requires_grad)
    # param.requires_grad = False
    break

best_loss=float('inf')
plot_loss=[]
# LDM.train()
# LDM.cond_stage_model.transformer.train()
dog_embeding=LDM.cond_stage_model.transformer.get_input_embeddings().weight[placeholder_token_ids[0]].data.clone()
david_embeding=LDM.cond_stage_model.transformer.get_input_embeddings().weight[placeholder_token_ids[1]].data.clone()

for ep in tqdm(range(epoch), total=epoch):
    LDM.model.diffusion_model.eval()
    LDM.first_stage_model.eval()
    # LDM.train()
    # LDM.cond_stage_model.transformer.train()
    LDM.cond_stage_model.transformer.text_model.embeddings.token_embedding.train()
    loss_list=[]
    for it, two_data in tqdm(enumerate(zip(dog_train_dataloader, david_train_dataloader)), total=max(len(dog_train_dataloader), len(david_train_dataloader))) :
        
        for id, data in enumerate(two_data):
            # step: Convert images to latent space
            
            latents= LDM.encode_first_stage(data["pixel_values"].to(dtype=weight_dtype).to(device))
            latents = LDM.get_first_stage_encoding(latents)
            # if DEBUG_FLAG: print(latents.shape)

            # step: Sample noise that we'll add to the latents
            noise = torch.randn_like(latents).to(device) 
            target = noise
            bsz = latents.shape[0]
            # if DEBUG_FLAG: print(latents.shape)
            # if DEBUG_FLAG: print(bsz)
            
            # step: Sample a random timestep for each image
            timesteps = torch.randint(0, num_train_timesteps, (bsz,), device=device).long()

            # step: Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = LDM.q_sample(x_start=latents, t=timesteps, noise=noise)
            # if DEBUG_FLAG: print(noisy_latents.shape)
            

            # step: Get the text embedding for conditioning
            encoder_hidden_states = LDM.cond_stage_model.transformer(data["input_ids"].to(device))[0].to(dtype=weight_dtype) 
            # if DEBUG_FLAG: print(encoder_hidden_states.shape)
            # if DEBUG_FLAG: break
            
            # step: Predict the noise residual
            model_pred = LDM.model.diffusion_model(x=noisy_latents, timesteps=timesteps, context=encoder_hidden_states)
            # model_pred = model_pred.sample

            # step: loss 
            # if DEBUG_FLAG: print(target.shape)
            loss = criterion(model_pred, target)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        
            # if DEBUG_FLAG: print(f"{loss.requires_grad=},{loss.grad=}")
            # if DEBUG_FLAG: print(f"{loss.requires_grad=},{loss.grad=}")
            # if DEBUG_FLAG: print(f"{loss.requires_grad=},{loss.grad=}")
            
            if DEBUG_FLAG: print(f"========log========")
            if DEBUG_FLAG: print(f"train on {id=}")
            if DEBUG_FLAG: print(f"dog embeding not changed: {LDM.cond_stage_model.transformer.get_input_embeddings().weight[placeholder_token_ids[0]].data[:5]==dog_embeding[:5]}")
            if DEBUG_FLAG: print(f"david embeding not changed: {LDM.cond_stage_model.transformer.get_input_embeddings().weight[placeholder_token_ids[1]].data[:5]==david_embeding[:5]}")

            # if DEBUG_FLAG: print(f"before: {LDM.cond_stage_model.transformer.get_input_embeddings().weight[placeholder_token_ids[0]][:5]=}")
            # if DEBUG_FLAG: print(f"before: {LDM.cond_stage_model.transformer.get_input_embeddings().weight[placeholder_token_ids[1]][:5]=}")
            if id==0: 
                dog_embeding=LDM.cond_stage_model.transformer.get_input_embeddings().weight[placeholder_token_ids[0]].data.clone() # update dog embedding
                LDM.cond_stage_model.transformer.get_input_embeddings().weight[placeholder_token_ids[1]].data=david_embeding # freeze davide embedding
            else: 
                david_embeding=LDM.cond_stage_model.transformer.get_input_embeddings().weight[placeholder_token_ids[1]].data.clone()
                LDM.cond_stage_model.transformer.get_input_embeddings().weight[placeholder_token_ids[0]].data=dog_embeding
            
            # if DEBUG_FLAG: print(f"after: {LDM.cond_stage_model.transformer.get_input_embeddings().weight[placeholder_token_ids[0]][:5]=}")
            # if DEBUG_FLAG: print(f"after: {LDM.cond_stage_model.transformer.get_input_embeddings().weight[placeholder_token_ids[1]][:5]=}")
            if DEBUG_FLAG: print(f"dog embeding not changed: {LDM.cond_stage_model.transformer.get_input_embeddings().weight[placeholder_token_ids[0]].data[:5]==dog_embeding[:5]}")
            if DEBUG_FLAG: print(f"david embeding not changed: {LDM.cond_stage_model.transformer.get_input_embeddings().weight[placeholder_token_ids[1]].data[:5]==david_embeding[:5]}")

            # Let's make sure we don't update any embedding weights besides the newly added token
            with torch.no_grad():
                LDM.cond_stage_model.transformer.get_input_embeddings().weight[:placeholder_token_ids[0]] = orig_embeds_params[:placeholder_token_ids[0]]
                if DEBUG_FLAG: print(f"orig weight:{LDM.cond_stage_model.tokenizer.convert_ids_to_tokens(placeholder_token_ids[id])=}, {orig_embeds_params[placeholder_token_ids[id]][:10]=}")
                if DEBUG_FLAG: print(f"embedding weight:{LDM.cond_stage_model.tokenizer.convert_ids_to_tokens(placeholder_token_ids[id])=}, {token_embeds[placeholder_token_ids[id]][:10]=}")
                if DEBUG_FLAG: print(token_embeds == orig_embeds_params)
                if DEBUG_FLAG: print(token_embeds[-4:] == orig_embeds_params[-4:])

    scheduler.step()
    plot_loss.append(sum(loss_list)/len(loss_list))
    print(f"avg loss: {sum(loss_list)/len(loss_list)}")
    # step: save newly train model (modified embedding and tokenizer)
    if best_loss> sum(loss_list)/len(loss_list):
        print(best_loss)
        best_loss=sum(loss_list)/len(loss_list)
        print(best_loss)    
        if not DEBUG_FLAG: 
            name=f'two_token_ep{epoch}_lr{lr}.pth' if resume== None else f'resume_two_token_ep{epoch}_lr{lr}.pth'   
            torch.save(LDM, os.path.join(save_path,name))
            print(f"saving at {os.path.join(save_path,name)}")

    # if DEBUG_FLAG: print(f"orig weight:{placeholder_tokens[0]}, {orig_embeds_params[min(placeholder_token_ids)][:5]=}")
    # if DEBUG_FLAG: print(f"embedding weight:{placeholder_tokens[0]}, {token_embeds[min(placeholder_token_ids)][:5]=}")

if DEBUG_FLAG: print(f"trained embeddings{LDM.cond_stage_model.transformer.get_input_embeddings()}")
# if DEBUG_FLAG: print(sum(LDM.cond_stage_model.transformer.get_input_embeddings().weight == orig_embeds_params))
plt.plot(range(len(plot_loss)), plot_loss)
plt.savefig(f"p3_loss_fig_lr{lr}_ep{epoch}.png")

# if __name__=="__main__":

    # print("==========================================")
    # print("==============train on dog==============")
    # print("==========================================")
    # new_token=["<new1>"]
    # placeholder_tokens = new_token

    # # add dummy tokens for multi-vector
    # additional_tokens = []
    # for i in range(len(placeholder_tokens) ):
    #     additional_tokens.append(f"{placeholder_tokens[i]}_{i}")
    # placeholder_tokens += additional_tokens

    # # add new token to tokenizer
    # num_added_token=LDM.cond_stage_model.tokenizer.add_tokens(placeholder_tokens)
    # placeholder_token_ids = LDM.cond_stage_model.tokenizer.convert_tokens_to_ids(placeholder_tokens)
    # if DEBUG_FLAG: print(f"{placeholder_token_ids=}")

    # # resize the embedding for newly added tokens
    # if DEBUG_FLAG: print(f"{len(LDM.cond_stage_model.tokenizer)=}")
    # LDM.cond_stage_model.transformer.resize_token_embeddings(len(LDM.cond_stage_model.tokenizer))
    # token_embeds = LDM.cond_stage_model.transformer.get_input_embeddings().weight.data

    # # initialized token with related concept
    # initializer_token_id = LDM.cond_stage_model.tokenizer.encode(["drawing","",], add_special_tokens=False)

    # with torch.no_grad():
    #     for i, token_id in enumerate(placeholder_token_ids):
    #         # if DEBUG_FLAG: print(f"{token_id}: {initializer_token_id[i]}")
    #         token_embeds[token_id] = token_embeds[initializer_token_id[i]].clone()
    #         token_embeds[token_id].requires_grad=True
    # orig_embeds_params = token_embeds.clone()

    # if DEBUG_FLAG: print(f"before initialized weight: {token_embeds.shape=}")
    # if DEBUG_FLAG: print(f"before initialized weight: {token_embeds[min(placeholder_token_ids) : max(placeholder_token_ids)+1].shape=}")
    # if DEBUG_FLAG: print(f"before initialized weight: {token_embeds[min(placeholder_token_ids) : max(placeholder_token_ids)+1]=}")
    # if DEBUG_FLAG: print(f"after initialized weight: {token_embeds[min(placeholder_token_ids) : max(placeholder_token_ids)+1]=}")

    # if DEBUG_FLAG: print(f"{token_embeds.shape=}")
    # if DEBUG_FLAG: print(f"{placeholder_tokens=}")
    # if DEBUG_FLAG: print(f"{placeholder_token_ids=}")
    # if DEBUG_FLAG: print(f"{num_added_token=}")
    # if DEBUG_FLAG: print(f"{LDM.cond_stage_model.tokenizer.convert_ids_to_tokens(placeholder_token_ids[0])=}")
    # if DEBUG_FLAG: print(f"added embeddings{LDM.cond_stage_model.transformer.get_input_embeddings()}")

    # dog_dataset=TextualInversionDataset(
    #         data_root=dog_dir,
    #         tokenizer=LDM.cond_stage_model.tokenizer,
    #         # size=args.resolution,
    #         placeholder_token=(LDM.cond_stage_model.tokenizer.convert_ids_to_tokens(placeholder_token_ids[0])),
    #         repeats=repeats,
    #         learnable_property="object",
    #         center_crop=False,
    #         set="train",
    #     )
    # train_dataloader=DataLoader(dog_dataset, batch_size=batch_size, shuffle=True )
    # train()

    # print("==========================================")    
    # print("==============train on david==============")
    # print("==========================================")
    # new_token=["<new2>"]
    # placeholder_tokens = new_token

    # # add dummy tokens for multi-vector
    # additional_tokens = []
    # for i in range(len(placeholder_tokens) ):
    #     additional_tokens.append(f"{placeholder_tokens[i]}_{i}")
    # placeholder_tokens += additional_tokens

    # # add new token to tokenizer
    # num_added_token=LDM.cond_stage_model.tokenizer.add_tokens(placeholder_tokens)
    # placeholder_token_ids = LDM.cond_stage_model.tokenizer.convert_tokens_to_ids(placeholder_tokens)
    # if DEBUG_FLAG: print(f"{placeholder_token_ids=}")

    # # resize the embedding for newly added tokens
    # if DEBUG_FLAG: print(f"{len(LDM.cond_stage_model.tokenizer)=}")
    # LDM.cond_stage_model.transformer.resize_token_embeddings(len(LDM.cond_stage_model.tokenizer))
    # token_embeds = LDM.cond_stage_model.transformer.get_input_embeddings().weight.data

    # # initialized token with related concept
    # initializer_token_id = LDM.cond_stage_model.tokenizer.encode(["drawing","",], add_special_tokens=False)

    # with torch.no_grad():
    #     for i, token_id in enumerate(placeholder_token_ids):
    #         # if DEBUG_FLAG: print(f"{token_id}: {initializer_token_id[i]}")
    #         token_embeds[token_id] = token_embeds[initializer_token_id[i]].clone()
    #         token_embeds[token_id].requires_grad=True
    # orig_embeds_params = token_embeds.clone()

    # if DEBUG_FLAG: print(f"before initialized weight: {token_embeds.shape=}")
    # if DEBUG_FLAG: print(f"before initialized weight: {token_embeds[min(placeholder_token_ids) : max(placeholder_token_ids)+1].shape=}")
    # if DEBUG_FLAG: print(f"before initialized weight: {token_embeds[min(placeholder_token_ids) : max(placeholder_token_ids)+1]=}")
    # if DEBUG_FLAG: print(f"after initialized weight: {token_embeds[min(placeholder_token_ids) : max(placeholder_token_ids)+1]=}")

    # if DEBUG_FLAG: print(f"{token_embeds.shape=}")
    # if DEBUG_FLAG: print(f"{placeholder_tokens=}")
    # if DEBUG_FLAG: print(f"{placeholder_token_ids=}")
    # if DEBUG_FLAG: print(f"{num_added_token=}")
    # if DEBUG_FLAG: print(f"{LDM.cond_stage_model.tokenizer.convert_ids_to_tokens(placeholder_token_ids[0])=}")
    # if DEBUG_FLAG: print(f"added embeddings{LDM.cond_stage_model.transformer.get_input_embeddings()}")

    # david_dateset=TextualInversionDataset(
    #         data_root=David_Revoy_dir,
    #         tokenizer=LDM.cond_stage_model.tokenizer,
    #         # size=args.resolution,
    #         placeholder_token=(LDM.cond_stage_model.tokenizer.convert_ids_to_tokens(placeholder_token_ids[0])),
    #         repeats=repeats,
    #         learnable_property="style",
    #         center_crop=False,
    #         set="train",
    #     )
    
    # train_dataloader=DataLoader(david_dateset, batch_size=batch_size, shuffle=True )
    # train(id='dog_and_david')


