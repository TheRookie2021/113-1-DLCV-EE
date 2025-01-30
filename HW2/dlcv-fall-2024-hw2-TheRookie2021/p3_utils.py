import argparse, os, sys, glob
import torch
import numpy as np
from PIL import Image
import PIL
from tqdm import tqdm, trange
from torchvision.utils import make_grid
from omegaconf import OmegaConf
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
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
from torch.optim.lr_scheduler import LambdaLR

sys.path.append('stable-diffusion/ldm/models/diffusion')
from ddpm import DiffusionWrapper

import matplotlib.pyplot as plt

# if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
PIL_INTERPOLATION = {
    "linear": PIL.Image.Resampling.BILINEAR,
    "bilinear": PIL.Image.Resampling.BILINEAR,
    "bicubic": PIL.Image.Resampling.BICUBIC,
    "lanczos": PIL.Image.Resampling.LANCZOS,
    "nearest": PIL.Image.Resampling.NEAREST,
}
# else:
#     PIL_INTERPOLATION = {
#         "linear": PIL.Image.LINEAR,
#         "bilinear": PIL.Image.BILINEAR,
#         "bicubic": PIL.Image.BICUBIC,
#         "lanczos": PIL.Image.LANCZOS,
#         "nearest": PIL.Image.NEAREST,
#     }
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class Img_Txt_Dataset(Dataset):
    def __init__(self, input_dir, condition, transform=None, target_transform=None):
        self.input_dir =input_dir
        self.condition= condition
        self.input_imgs = [i for i in sorted(os.listdir(input_dir)) if i.endswith('png') or i.endswith('jpg')]
        self.transform = transform

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_imgs[idx])
        input_img = Image.open(input_path)
        
        if self.transform!=None:
            input_img = self.transform(input_img)
        # return input_img, self.condition
        return input_img 

    def __len__(self):
        return len(self.input_imgs)
class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        transform=None
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats
        # self.transform=transform
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)
        # print(placeholder_string, text)
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]


        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        
        # if self.transform != None:
        #     image=self.transform(image)
        
        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example

def train(LDM, device, new_token, initialized_concept, train_data_path, lr=0.001, epoch=30, num_train_timesteps=10000,
          save_path="model_ckpt/p3", DEBUG_FLAG=False):
    # new_token=["<new1>"]

    if DEBUG_FLAG: print("LDM components:")
    if DEBUG_FLAG: print(f"\t{LDM.__class__.__name__=}") # LatentDiffusion
    if DEBUG_FLAG: print(f"\t{LDM.model.__class__.__name__=}") # DiffusionWrapper
    if DEBUG_FLAG: print(f"\t{LDM.model.diffusion_model.__class__.__name__=}") # UNetModel
    if DEBUG_FLAG: print(f"\t{LDM.first_stage_model.__class__.__name__=}") # AutoencoderKL
    if DEBUG_FLAG: print(f"\t{LDM.cond_stage_model.__class__.__name__=}") # FrozenCLIPEmbedder
    if DEBUG_FLAG: print(f"\t{LDM.cond_stage_model.tokenizer.__class__.__name__=}") # CLIPTokenizer
    if DEBUG_FLAG: print(f"\t{LDM.cond_stage_model.transformer.__class__.__name__=}") # CLIPTextModel

    if not os.path.isdir(save_path): os.makedirs(save_path)
    placeholder_tokens = new_token

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(len(placeholder_tokens) ):
        additional_tokens.append(f"{placeholder_tokens[i]}_{i}")
    placeholder_tokens += additional_tokens

    # add new token to tokenizer
    num_added_token=LDM.cond_stage_model.tokenizer.add_tokens(placeholder_tokens)
    placeholder_token_ids = LDM.cond_stage_model.tokenizer.convert_tokens_to_ids(placeholder_tokens)
    if DEBUG_FLAG: print(f"{placeholder_token_ids=}")

    # resize the embedding for newly added tokens
    if DEBUG_FLAG: print(f"{len(LDM.cond_stage_model.tokenizer)=}")
    LDM.cond_stage_model.transformer.resize_token_embeddings(len(LDM.cond_stage_model.tokenizer))
    token_embeds = LDM.cond_stage_model.transformer.get_input_embeddings().weight.data

    # initialized token with related concept
    initializer_token_id = LDM.cond_stage_model.tokenizer.encode(["dog",""], add_special_tokens=False)

    with torch.no_grad():
        for i, token_id in enumerate(placeholder_token_ids):
            # if DEBUG_FLAG: print(f"{token_id}: {initializer_token_id[i]}")
            token_embeds[token_id] = token_embeds[initializer_token_id[i]].clone()
            token_embeds[token_id].requires_grad=True
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

    dataset=TextualInversionDataset(
            data_root=train_data_path,
            tokenizer=LDM.cond_stage_model.tokenizer,
            # size=args.resolution,
            placeholder_token=(LDM.cond_stage_model.tokenizer.convert_ids_to_tokens(placeholder_token_ids[0])),
            repeats=20,
            learnable_property="object",
            center_crop=False,
            set="train",
    )

    # david_dateset=TextualInversionDataset(
    #         data_root=David_Revoy_dir,
    #         tokenizer=LDM.cond_stage_model.tokenizer,
    #         # size=args.resolution,
    #         placeholder_token=(LDM.cond_stage_model.tokenizer.convert_ids_to_tokens(placeholder_token_ids[1])),
    #         repeats=20,
    #         learnable_property="style",
    #         center_crop=False,
    #         set="train",
    #     )
    # train_datasets=torch.utils.data.ConcatDataset([dog_dataset, david_dateset])
    # if DEBUG_FLAG: print(f"{len(train_datasets)=}")
    train_dataloader=DataLoader(dataset, batch_size=4, shuffle=True )
    if DEBUG_FLAG: print(f"{len(train_dataloader)=}")
    print(f"{len(train_dataloader)=}")



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
    lambda1 = lambda epoch: 0.2 * epoch if epoch > 5 else 1
    lambda2 = lambda epoch: 0.2 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)

    criterion = torch.nn.MSELoss(reduction="mean")
    # ====freeze====
    # LDM.model.diffusion_model.requires_grad=False
    # LDM.first_stage_model.requires_grad=False
    # LDM.cond_stage_model.transformer.text_model.encoder.requires_grad=False
    # LDM.cond_stage_model.transformer.text_model.final_layer_norm.requires_grad=False
    # LDM.cond_stage_model.transformer.text_model.embeddings.position_embedding.requires_grad=False
    # LDM.cond_stage_model.transformer.text_model.embeddings.token_embedding.requires_grad=True


    # for param in LDM.model.diffusion_model.parameters():
    #     param.requires_grad = False
    # for param in LDM.first_stage_model.parameters():
    #     param.requires_grad = False
    LDM.model.diffusion_model.eval()
    LDM.first_stage_model.eval()

    LDM.cond_stage_model.transformer.text_model.encoder.requires_grad_(False)
    LDM.cond_stage_model.transformer.text_model.final_layer_norm.requires_grad_(False)
    LDM.cond_stage_model.transformer.text_model.embeddings.position_embedding.requires_grad_(False)
    LDM.cond_stage_model.transformer.text_model.embeddings.token_embedding.requires_grad_(True)
    # LDM.cond_stage_model.transformer.get_input_embeddings().requires_grad_(True)



    # if DEBUG_FLAG: print(f"{LDM.cond_stage_model.transformer.text_model.embeddings.token_embedding.requires_grad=}")
    if DEBUG_FLAG: print(orig_embeds_params.shape)
    index_no_updates = torch.ones((len(LDM.cond_stage_model.tokenizer),), dtype=torch.bool)
    index_no_updates[min(placeholder_token_ids) :  max(placeholder_token_ids) + 1] = False


    # LDM.cond_stage_model.transformer.text_model.embeddings.token_embedding.train()
    # for param in LDM.cond_stage_model.transformer.parameters():
    #     print(param.requires_grad)
    #     # param.requires_grad = False
    #     break

    best_loss=float('inf')
    plot_loss=[]
    # LDM.train()
    # LDM.cond_stage_model.transformer.train()
    for ep in tqdm(range(epoch), total=epoch):
        LDM.model.diffusion_model.eval()
        LDM.first_stage_model.eval()
        # LDM.train()
        # LDM.cond_stage_model.transformer.train()
        LDM.cond_stage_model.transformer.text_model.embeddings.token_embedding.train()
        loss_list=[]
        for it, data in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
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
            # noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            noisy_latents = LDM.q_sample(x_start=latents, t=timesteps, noise=noise)

            # if DEBUG_FLAG: print(noisy_latents.shape)
            
            # step: Get the text embedding for conditioning
            temp=data["input_ids"]
            if DEBUG_FLAG: print(f"{temp=}")
            # encoder_hidden_states = LDM.cond_stage_model.transformer(data["input_ids"].to(device))[0].to(dtype=weight_dtype) 
            encoder_hidden_states = LDM.cond_stage_model.transformer(data["input_ids"].to(device))
            encoder_hidden_states = encoder_hidden_states[0].to(dtype=weight_dtype) 
            if DEBUG_FLAG: print(encoder_hidden_states.shape)
            
            # step: Predict the noise residual
            model_pred = LDM.model.diffusion_model(x=noisy_latents, timesteps=timesteps, context=encoder_hidden_states)
            # model_pred = model_pred.sample
            

            # step: loss 
            # if DEBUG_FLAG: print(target.shape)
            # loss = criterion(model_pred, target)
            loss = torch.nn.functional.mse_loss((model_pred).float(), (target).float(), reduction="mean")                  

            if DEBUG_FLAG: print(loss)
            loss_list.append(loss.item())
            

            loss.backward()
            if DEBUG_FLAG: print(f"{loss.requires_grad=},{loss.grad=}")
            optimizer.step()
            if DEBUG_FLAG: print(f"{loss.requires_grad=},{loss.grad=}")
            scheduler.step()
            if DEBUG_FLAG: print(f"{loss.requires_grad=},{loss.grad=}")
            optimizer.zero_grad()
            
            # Let's make sure we don't update any embedding weights besides the newly added token
            with torch.no_grad():
                LDM.cond_stage_model.transformer.get_input_embeddings().weight[:placeholder_token_ids[0]] = orig_embeds_params[:placeholder_token_ids[0]]
                if DEBUG_FLAG: print(f"orig weight:{LDM.cond_stage_model.tokenizer.convert_ids_to_tokens(placeholder_token_ids[0])=}, {orig_embeds_params[placeholder_token_ids[0]][:10]=}")
                if DEBUG_FLAG: print(f"embedding weight:{LDM.cond_stage_model.tokenizer.convert_ids_to_tokens(placeholder_token_ids[0])=}, {token_embeds[placeholder_token_ids[0]][:10]=}")
                if DEBUG_FLAG: print(token_embeds == orig_embeds_params)
                if DEBUG_FLAG: print(token_embeds[-4:] == orig_embeds_params[-4:])

        plot_loss.append(sum(loss_list)/len(loss_list))
        print(f"avg loss: {sum(loss_list)/len(loss_list)}")
        # step: save newly train model (modified embedding and tokenizer)
        if best_loss> sum(loss_list)/len(loss_list):
            print(best_loss)
            best_loss=sum(loss_list)/len(loss_list)
            print(best_loss)    
            if not DEBUG_FLAG: 
                name=f'{initialized_concept}_ep{epoch}_lr{lr}.pth'
                torch.save(LDM, os.path.join(save_path,name))
                print(f"saving at {os.path.join(save_path,name)}")

        # if DEBUG_FLAG: print(f"orig weight:{placeholder_tokens[0]}, {orig_embeds_params[min(placeholder_token_ids)][:5]=}")
        # if DEBUG_FLAG: print(f"embedding weight:{placeholder_tokens[0]}, {token_embeds[min(placeholder_token_ids)][:5]=}")

    if DEBUG_FLAG: print(f"trained embeddings{LDM.cond_stage_model.transformer.get_input_embeddings()}")
    # if DEBUG_FLAG: print(sum(LDM.cond_stage_model.transformer.get_input_embeddings().weight == orig_embeds_params))
    plt.plot(range(len(plot_loss)), plot_loss)
    plt.savefig(f"p3_{initialized_concept}_loss_fig_lr{lr}_ep{epoch}.png")
