import os
import torch
import json
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoProcessor, LlavaForConditionalGeneration
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from PIL import Image

DEBUG_MODE=True
def conver_to_template(conversation):
    """    
    from: conversation={'role': 'user', 'content': [{'type': 'text', 'text': 'Please give me a one sentence caption about the image Please give me a one sentence caption about the image Please give me a one sentence caption about the image Please give me a one sentence caption about the image.'}, {'type': 'image'}]}
    to: prompt= 'USER: <image>\nPlease give me a one sentence caption about the image. ASSISTANT:'
    """    
    prompt=conversation['content'][0]['text']
    template=f'USER: <image>\n{prompt} ASSISTANT:'
    return template

def main(data_dir, save_json_path):
    # step: prepare input output model
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    filepaths=[os.path.join(data_dir, name) for name in sorted(os.listdir(data_dir)) ]
    model_id = "llava-hf/llava-1.5-7b-hf"
    save_folder="p3_llava"
    os.makedirs(save_folder, exist_ok=True)
            
    caption_dict={}

    # step: load model
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        load_in_4bit=True
    ).to(0)
    processor = AutoProcessor.from_pretrained(model_id)
    # LlamaDecoderLayer
    # print(model)
    # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image") 
    conversations = [    
        # best: CIDEr: 1.1604811812238933 | CLIPScore: 0.77781494140625
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please give me a one sentence caption about the image."},
            {"type": "image"},
            ],
        },    
    ]
    
    for conversation in conversations:
        print(f"{conversation=}")
        prompt = conver_to_template(conversation)
        print(f"{prompt=}")
        # break
        if DEBUG_MODE: print("======== start inferencing ========")
        if DEBUG_MODE: print(f"{prompt=}")
        for img_p in tqdm(filepaths, total=len(filepaths)):
            # print(f"======== {img_p} ========")
            image = Image.open(img_p)
            inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)
            # print(inputs )
            # attention_mask=[]
            cap_output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            output= model(**inputs, output_attentions=True, output_hidden_states=True, use_cache=False)
            print(inputs['pixel_values'].shape)
            # model.get_attention_map
            # print(inputs['images'].shape)
            # print(type(output.attentions)) # torch.Size([1, 32, 600, 600])
            # print(len(output.attentions))
            # print(output_hidden_states)    
            # print(output.attentions.shape)    
            # att_map=output.attentions[-1][:,:,-1,:576]
            # print(att_map)
            
            """
            attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — 
            Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
            """
            # print(cap_output)
            print(cap_output.shape) # torch.Size([1, 613])
            caption=processor.decode(cap_output[0][2:], skip_special_tokens=True)
            print(caption)


            # ====
            ori_im = Image.open(img_p)

            # plt.subplot(2, len(output.attentions)//2+2, 1)
            # plt.imshow(ori_im)
            # plt.title("start")
            # plt.axis(False)    
            tokens=caption.split(':')[-1].split(" ")
            print(tokens)
            # for id, att_map in enumerate(output.attentions):
            #     plt.subplot(2, len(output.attentions)//2+2, id+2)
            #     att_map=att_map[:,:,-1,:576]
            #     print(att_map.shape) 
            #     # break
            #     # note: decrease channel from 12 to 1
            #     att_map= torch.mean(att_map, dim=1) 
            #     att_map -= torch.min(att_map)
            #     att_map /= torch.max(att_map)
                
            #     att_map=att_map.view(1, 24, 24)
            #     att_map=torch.nn.functional.interpolate(att_map.unsqueeze(0),size=(336,336), mode="bilinear", align_corners=False)
            #     att_map=att_map.squeeze().detach().cpu().numpy()
            #     # step: start ploting the graph
            #     # word=[tokens[id]]
            #     plt.imshow(ori_im)
            #     plt.imshow(att_map, cmap='jet', alpha=0.3)
            #     # plt.title(word)
            #     plt.axis(False)
            # plt.savefig(os.path.join(save_folder,f"all.png"))
            # ====
            # ori_im = Image.open(img_p)
            # ori_im = ori_im.resize((336,336), Image.ANTIALIAS)
            # final_att_sum=output.attentions[0]
            # print(len(output.attentions))
            # ====TO find out  32 means
            # for i, map in enumerate(output.attentions):
            #     final_att_sum+= map
            #     map=map[:,:,-1,1:577]
            #     # plt.subplot(2, len(output.attentions)//2, i+1)
            #     # map=map[:,-1: 576]
                
            #     # note: decrease channel from 12 to 1
            #     # map= torch.mean(map, dim=1)
            #     # map -= torch.min(map)
            #     # map /= torch.max(map)
            #     map= map[-1,:,:] 
            #     map= torch.norm(map, dim=0) 
            
            #     map=map.view(1, 24, 24)
            #     print(map) 
            #     print() 
            #     map=torch.nn.functional.interpolate(map.unsqueeze(0),size=(336,336), mode="bilinear", align_corners=False)
                
            #     # one_att_map=one_att_map.view(1, 24, 24)
            #     # one_att_map=torch.nn.functional.interpolate(one_att_map.unsqueeze(0),size=(336,336), mode="bilinear", align_corners=False)
            #     # one_att_map=one_att_map.squeeze().detach().cpu().numpy()
            #     # step: start ploting the graph
            #     # word=[tokens[id]]
            #     plt.imshow(ori_im)
            #     plt.imshow(map.squeeze().detach().cpu().numpy(), cmap='jet', alpha=0.3)
            #     # plt.title(word)
            #     plt.axis(False)
            #     plt.colorbar()
            #     plt.savefig(os.path.join(save_folder,f"{i}_check32.png"))
            #     plt.clf()
            # final_att_sum=final_att_sum[:,:,-1,1:577]
                
            # final_att_sum= torch.mean(final_att_sum, dim=1)
            # final_att_sum -= torch.min(final_att_sum)
            # final_att_sum /= torch.max(final_att_sum)
            # final_att_sum=final_att_sum.view(1, 24, 24)
            # final_att_sum=torch.nn.functional.interpolate(final_att_sum.unsqueeze(0),size=(336,336), mode="bilinear", align_corners=False)

            # print(final_att_sum) 
            print() 
            
            # one_att_map=one_att_map.view(1, 24, 24)
            # one_att_map=torch.nn.functional.interpolate(one_att_map.unsqueeze(0),size=(336,336), mode="bilinear", align_corners=False)
            # one_att_map=one_att_map.squeeze().detach().cpu().numpy()
            # step: start ploting the graph
            # word=[tokens[id]]
            # plt.imshow(ori_im)
            # plt.imshow(final_att_sum.squeeze().detach().cpu().numpy(), cmap='jet')
            # plt.title(word)
            
            # plt.imshow(ori_im)
            # plt.axis(False)
            # plt.colorbar()
            # plt.savefig(os.path.join(save_folder,f"sum_check32.png"))
            plt.clf()
            # break
            # ====TO find out  32 means








            ori_im = Image.open(img_p)
            ori_im = ori_im.resize((336,336), Image.ANTIALIAS)
            
            att_map=output.attentions[1]
            # plt.subplot(2, len(output.attentions)//2+1, 1)
            num_of_word=len(tokens)-1
            for i, word in enumerate(tokens):
                one_att_map=att_map[:,:,-num_of_word+i,1:577]
                # print(att_map.shape) 
                plt.subplot(2, len(tokens)//2+1, i+1)
                
                # note: decrease channel from 12 to 1
                one_att_map= torch.mean(one_att_map, dim=1) 
                # one_att_map= one_att_map[-1,:,:] 
                one_att_map= torch.norm(one_att_map, dim=0) 
            
                # att_map -= torch.min(att_map)
                # att_map /= torch.max(att_map)
                
                one_att_map=one_att_map.view(1, 24, 24)
                one_att_map=torch.nn.functional.interpolate(one_att_map.unsqueeze(0),size=(336,336), mode="bilinear", align_corners=False)
                one_att_map=one_att_map.squeeze().detach().cpu().numpy()
                # step: start ploting the graph
                # word=[tokens[id]]
                if i==0:
                    plt.imshow(ori_im)
                    plt.title("start")
                    plt.axis(False)   
                else:
                    plt.imshow(ori_im)
                    plt.imshow(one_att_map, cmap='jet', alpha=0.5)
                    plt.title(word)
                    plt.axis(False)
            plt.savefig(os.path.join(save_folder,f"{os.path.basename(img_p)}"))
            # TODO: 
                # torch.Size([1, 32, 600, 600])
                # torch.Size([1, 32, -1, :576])
                # torch.Size([1, 32, -2, :576])
            # print(output[0])
            # print(output[1])
            # break
            # step: parse output 
            # print(f"{os.path.basename(img_p).split('.')[0]}, {caption.split(':')[-1]}")
            # caption_dict[os.path.basename(img_p).split('.')[0]]=caption.split(':')[-1]

        # print(caption_dict)
        # # final_save_path=fileNamer(save_json_path)
        # # print(f"save at: {final_save_path=}")
        # # with open(final_save_path, 'w') as f:
        #     json.dump(caption_dict, f)





if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data_root", default="hw3_data/p3_data/images", help="Image root")
    parser.add_argument("--save_json_path", default="p1/prediction.json", help="output json file")

    args = parser.parse_args()
    main(data_dir=args.data_root, save_json_path=args.save_json_path)