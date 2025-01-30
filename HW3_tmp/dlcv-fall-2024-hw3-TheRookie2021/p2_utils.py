# for dataset
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms  as transforms
from PIL import Image
import os
import json
import numpy as np
# for model
import torch
from torch import Tensor, nn
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import timm
import loralib as lora
from p2_decoder import Decoder, Config

from matplotlib import pyplot as plt
import matplotlib.cm as cm

DEBUG=False
EOS_TOKEN=50256
BOS_TOKEN=50256
TESTING_PAD_TOKEN=BOS_TOKEN
def plot_loss_graph(checkpoint_folder, lossA, lossB, labelA, labelB ):
    if os.path.isdir(checkpoint_folder):
        save_path=os.path.join(checkpoint_folder, f"train_loss_plot.png")
    else:
        save_path=os.path.join(os.pardir(checkpoint_folder), f"train_loss_plot.png")
    
    print(f"loss fig: {save_path}")
    plt.plot(range(len(lossA)), lossA, label=labelA )  # Plot the chart
    plt.plot(range(len(lossB)), lossB, label=labelB )  # Plot the chart
    plt.savefig(save_path)  

def parseJSON(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
        
        annotation=data['annotations']
        annotation=[(pair['image_id'], pair['caption']) for pair  in annotation ]
        annotation=sorted(annotation)

        images=data['images']
        # print(len(images))
        image_id_mapping={}
        for pair  in images:
            image_id_mapping[pair['id']]= pair['file_name']

        return annotation, image_id_mapping
    except:
        if DEBUG: print("error while reading json file")
        return None

def pad_tokens(input, max_len=100):
    if max_len==0:
        return []
    if len(input)>max_len: 
        input[-1]=50256 # endoftext token id
        return input[:max_len]
    max_seq=[50256]*max_len
    output=input+max_seq[len(input):]
    return output

def random_mask(input, p=0.2):
    output=np.array(input)
    indices = np.random.choice(np.arange(output.size), replace=False,
                           size=int(output.size * p))
    output[indices] = 50256
    return list(output)

def trim_tokens(input):
    # print(input)
    input=input[1:]
    for index, j in enumerate(input):
        if j==50256: return input[:index]
    
    return input[1:]
# TODO: dataset format is different from P1
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, annotaion_path, prompt="Please give me a one sentence caption about the image.", max_len=100, transform=None, tokenizer=None,):
        self.img_dir = img_dir
        self.prompt=prompt
        # self.filepath=[os.path.join(img_dir, name) for name in sorted(os.listdir(img_dir)) ]
        self.captions, self.image_id_mapping = parseJSON(annotaion_path) # note: parse json file into a sorted list 
        self.tokenizer = tokenizer # note: tokenize input string
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        (id, caption)= self.captions[idx]
        name=self.image_id_mapping[id]
        caption=caption.capitalize()
        img_path=os.path.join(self.img_dir, name)
        # print(caption, id, name, img_path)
        # img_path = self.filepath[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            if DEBUG: print(img_path)

        # TODO: check the label format to be returned  
        gt_label=caption
        if self.tokenizer:
            # gt_label= f"<|endoftext|>{gt_label}<|endoftext|>"
            # prompt=f"<|endoftext|>{self.prompt}<|endoftext|>"
            gt_label = self.tokenizer.encode(gt_label, allowed_special="<|endoftext|>")
            prompt=self.tokenizer.encode(self.prompt, allowed_special="<|endoftext|>")
            
            gt_label= gt_label + [50256]
            masked_label=[50256] + gt_label.copy()
            prompt= [50256] + prompt+ [50256] 
            # print(gt_label)
            # print(f"{len(gt_label)=}")
            # gt_label=pad_tokens(gt_label, max_len=self.max_len)
            # masked_label=random_mask(gt_label, 0.2)
            # masked_label=pad_tokens(masked_label, max_len=self.max_len)
            # prompt=pad_tokens(prompt, max_len=self.max_len)
        if DEBUG: print("info:",image.dtype, torch.tensor(prompt).dtype ,torch.tensor(gt_label).dtype)
        if DEBUG: print("shape:",image.shape, torch.tensor(prompt).shape ,torch.tensor(gt_label).shape)
        if DEBUG: print("shape:",image.shape, torch.tensor(prompt).shape ,torch.tensor(gt_label).shape, torch.tensor(masked_label).shape)
        data={
            "image":image, 
            "prompt":torch.tensor(prompt) ,
            "gt_label":torch.tensor(gt_label), 
            "img_path":img_path, 
            "masked_label":torch.tensor(masked_label),
            "len":int(len(gt_label))
        }
        # return image, torch.tensor(prompt) ,torch.tensor(gt_label), img_path, torch.tensor(masked_label),
        return data    
    def collate_fn_padd(self, batch):
        '''
        Padds batch of variable length

        note: it converts things ToTensor manually here since the ToTensor transform
        assume it takes in images rather than arbitrary tensors.
        '''
        result_dict={}
        # step: to collect a batch of data into one dictionay 
        for dict in batch:
            for key in dict:
                # print(result_dict)
                if key not in result_dict:
                    result_dict[key]=[dict[key]]
                else:
                    result_dict[key].append( dict[key]) 
        # padd input prompt
        
        result_dict["image"]= torch.stack( result_dict["image"], dim=0)
        result_dict['prompt']=    torch.nn.utils.rnn.pad_sequence(result_dict['prompt'], padding_value=TESTING_PAD_TOKEN).swapaxes(0,1)
        result_dict['masked_label']=    torch.nn.utils.rnn.pad_sequence(result_dict['masked_label'], padding_value=TESTING_PAD_TOKEN).swapaxes(0,1)

        return result_dict
# =========================================================================================
# note: Model with Vit encoder
class VitCaptionModel(torch.nn.Module):
    """
    ## Input:
    - pre-processed RGB image
    - pre-processed caption string
    ## output:
    - string: caption
    ## Ingredients:
    - pretrained vision encoder: ViT, encoder image into embeddings
    - pretrained text decoder: provided by TA, need to be extended (PEFT) 
    - tokenizer: provided by TA
        - to convert input prompt into embeddings, e.g. prompt= "what is in the image?" 
        - to convert captions(ground-truth label) into embeddings for calculating loss, e.g "'Athletes sitting on bench at sideline during game.'"
    ## Task: train the model by PEFT, i.e. Lora
        - only add PEFT on decoder, i.e. freeze LLM
    
    """
    def __init__(self, decoder_ckpt,  train_lora=True, lora_rank=0):
        super(VitCaptionModel, self).__init__()
        # note: Vision encoder, https://huggingface.co/timm/vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k
        self.vision_encoder = timm.create_model(
            #! check version of timm and corrresponding model availble
                                # 'vit_betwixt_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k',
                                "vit_large_patch14_clip_224.openai_ft_in12k_in1k",
                                # "vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k",
                                pretrained=True,
                                num_classes=0,  # remove classifier nn.Linear
                            )
        
        # note: to project image embedding to word embedding
        self.img_projection_layer=nn.Linear(1024, 768)

        # note: for image input transform
        self.data_config = timm.data.resolve_model_data_config(self.vision_encoder)
        
        # note: Text decoder
        # note: Lora layer, https://github.com/microsoft/LoRA
        self.decoder_config=Config(decoder_ckpt, lora_attn_dim=lora_rank)
        self.decoder_config.train_lora=train_lora
        self.text_decoder=Decoder(self.decoder_config) 
        self.softmax=nn.Softmax(dim=-1)
    
    def forward(self, imgs, token,):
        # step: deal with a batch of images
        img_embedding = self.vision_encoder.forward_features(imgs)  # [Batch, 145, 1024]
        # print("feature shape: ",img_embedding.shape)
        
        # step: project image embedding to text embedding
        img_embedding = self.img_projection_layer(img_embedding)

        # step: deal with a batch of captions
        if DEBUG: print(token.dtype, img_embedding.dtype)
        decoder_output, att_map = self.text_decoder(token=token,  img_embedding=img_embedding)
        if DEBUG: print(f"{decoder_output.dtype=}")

        return decoder_output
    
    @torch.no_grad()
    def inference(self, img_embedding, max_length=30):
        self.eval()
        device=img_embedding.device
        y_input = torch.tensor([[EOS_TOKEN]], dtype=torch.long, device=device)

        # num_tokens = len(img_embedding[0])

        for _ in range(max_length):
            # print(y_input)
            pred= self.forward(img_embedding, y_input)
            # num with highest probability
            next_item = pred.topk(1)[1].view(-1)[-1].item() 
            next_item = torch.tensor([[next_item]], device=device)
            # print(next_item)
            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)
            # Stop if model predicts end of sentence
            if next_item.view(-1).item() == EOS_TOKEN:
                break
        return y_input.view(-1).tolist()
    
    def visualize(self, img, img_path, tokenizer,max_length=15):
        def expend_mask(mask, template):
            # print(mask.shape)
            # note: assume mask and template are square
            template=nn.functional.interpolate(mask.unsqueeze(0),size=(224,224), mode="bilinear", align_corners=False)
            return template.squeeze().cpu().numpy()

        self.eval()
        device=img.device
        y_input = torch.tensor([[EOS_TOKEN]], dtype=torch.long, device=device)
        img_embedding = self.vision_encoder.forward_features(img)  # [Batch, 145, 1024]
        img_embedding = self.img_projection_layer(img_embedding) # step: project image embedding to text embedding
        print(img_embedding.shape[1])
        size=img_embedding.shape[1]

        list_att_map=[]
        for id in range(max_length):
            
            pred, att_map = self.text_decoder(img_embedding=img_embedding, token=y_input)
            # print(att_map.shape) 
            att_map= att_map[:,:,-1,0:size-1]
            
            # note: decrease channel from 12 to 1
            # att_map= torch.mean(att_map, dim=1)
            # att_map= att_map[-1,:,:] 
            # att_map= torch.norm(att_map, dim=0) 
            
            # att_map -= torch.min(att_map)
            # att_map /= torch.max(att_map)
            # att_map*=255
            # print(att_map.to(torch.int32))
            att_map=att_map.view(1, 16, 16)
            list_att_map.append(att_map)
           
            # greedyly get the best prediction
            next_item = pred.topk(1)[1].view(-1)[-1].item() # num with highest probability
            next_item = torch.tensor([[next_item]], device=device)
            
            # Concatenate previous input with predicted best word
            y_input = torch.cat((y_input, next_item), dim=1)
            if next_item.view(-1).item() == EOS_TOKEN:
                break

        tokens=y_input.view(-1).tolist()
        ori_im = Image.open(img_path)
        ori_im = ori_im.resize((224,224), Image.ANTIALIAS)
        template=torch.zeros((224,224))
        
        
        sub_folder=os.path.basename(img_path).split(".")[0]
        save_folder=os.path.join("test_attmap", sub_folder)
        tokens=tokens[1:]
        os.makedirs(save_folder, exist_ok=True)
        
        # step: start ploting the graph
        plt.subplot(2, len(list_att_map)//2+1, 1)
        plt.imshow(ori_im)
        plt.title("start")
        plt.axis(False)    
        for id, map in enumerate(list_att_map):
            plt.subplot(2, len(list_att_map)//2+1, id+2)
            # TODO: combine with the img
            word=tokenizer.decode([tokens[id]])
            # TODO: resize
            mask=expend_mask(map, template)
            plt.imshow(ori_im)
            plt.imshow(mask, cmap='jet', alpha=0.3)
            plt.title(word)
            plt.axis(False)
        plt.savefig(os.path.join(save_folder,f"all.png"))
        print(os.path.join(save_folder,f"all.png"))
        return y_input.view(-1).tolist()
    
    def beam_search_ori(self, img, beams=3, max_length=30):
        def get_pred(x: Tensor, encoder_feature: Tensor):
            x = torch.narrow(x, 1, 0, min(x.size(1), self.text_decoder.block_size))
            pos = torch.arange(
                x.size()[1], dtype=torch.long, device=x.device
            ).unsqueeze(0)
            x = self.text_decoder.transformer.wte(x) + self.text_decoder.transformer.wpe(pos)
            concated_embedding= torch.concat((encoder_feature, x, ), dim=1 )# torch.Size([4, 195, 768])
            for block in self.text_decoder.transformer.h:
                concated_embedding, att= block(concated_embedding)
            x = self.text_decoder.lm_head(self.text_decoder.transformer.ln_f(concated_embedding[:, -1, :]))
            # print(x.shape)
            return x

        self.eval()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # step: dealing with image encoding
        encoder_feature = self.vision_encoder.forward_features(img)
        encoder_feature = self.img_projection_layer(encoder_feature)
        cur_state = torch.tensor([[EOS_TOKEN]]).to(device)
        
        # step: beam search
        # get top k words
        next_probs = get_pred(cur_state, encoder_feature)

        vocab_size = next_probs.shape[-1]

        # probs, pred id
        cur_probs, next_chars = next_probs.log_softmax(-1).topk(k=beams, axis=-1)
        cur_probs = cur_probs.reshape(beams)
        next_chars = next_chars.reshape(beams, 1)
        # gen first k beams
        cur_state = cur_state.repeat((beams, 1))  
        cur_state = torch.cat((cur_state, next_chars), axis=1)

        ans_ids = []
        ans_probs = []
        for i in range(1, max_length):
            # get top k beams for beam*beam candidates
            next_probs = get_pred(
                cur_state, encoder_feature.repeat((beams, 1, 1))
            ).log_softmax(-1)
            cur_probs = cur_probs.unsqueeze(-1) + next_probs
            cur_probs = cur_probs.flatten()  

            # length normalization
            _, idx = (cur_probs / (len(cur_state[0]) + 1)).topk(k=beams, dim=-1)
            cur_probs = cur_probs[idx]

            # get corresponding next char
            next_chars = torch.remainder(idx, vocab_size)
            next_chars = next_chars.unsqueeze(-1)
            # print("next char: ",next_chars)

            # get corresponding original beams
            top_candidates = (idx / vocab_size).long()  
            cur_state = cur_state[top_candidates]
            cur_state = torch.cat((cur_state, next_chars), dim=1)

            # concat next_char to beams
            to_rm_idx = set()
            for idx, ch in enumerate(next_chars):
                if i == (max_length - 2) or ch.item() == EOS_TOKEN:
                    ans_ids.append(cur_state[idx].cpu().tolist())
                    ans_probs.append(cur_probs[idx].item() / len(ans_ids[-1]))
                    to_rm_idx.add(idx)
                    beams -= 1

            to_keep_idx = [i for i in range(len(cur_state)) if i not in to_rm_idx]
            if len(to_keep_idx) == 0:
                break
            cur_state = cur_state[to_keep_idx]
            cur_probs = cur_probs[to_keep_idx]
        max_idx = torch.argmax(torch.tensor(ans_probs)).item()
        return ans_ids[max_idx]
    

if __name__=='__main__':
    pass
    # test: json parser
    # json_file="hw3_data/p2_data/train.json"
    # parseJSON(json_file)

    # test: Dataset
    # from tokenizer import BPETokenizer

    # img_dir='hw3_data/p2_data/images'
    # annotation_path='hw3_data/p2_data'
    # checkpoint_folder="p2_models"
    # model= VitCaptionModel(decoder_ckpt="hw3_data/p2_data/decoder_model.bin", lora_rank=16)
    # tokenizer=BPETokenizer("encoder.json", "vocab.bpe")
    # TRANSFORM= timm.data.create_transform(**model.data_config, is_training=False)
    # print(TRANSFORM)
    # train_dataset=  CustomImageDataset(img_dir=os.path.join(img_dir, "train"), 
    #                                 annotaion_path=os.path.join(annotation_path,'train.json'), 
    #                                 transform=TRANSFORM,
    #                                 tokenizer=tokenizer)
    # dataset=   CustomImageDataset(img_dir=os.path.join(img_dir, "val"), 
    #                                 annotaion_path=os.path.join(annotation_path, 'val.json'), 
    #                                 transform=TRANSFORM,
    #                                 tokenizer=tokenizer)
    # train_dataloader=   DataLoader(train_dataset, batch_size=1, collate_fn=train_dataset.collate_fn_padd, shuffle=True)
    # from tqdm import tqdm
    # for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
    #     img_path=data['img_path']        
    #     if i==5:break

    # print(len(train_dataset))

    # test: Model structure

