import requests
from PIL import Image
import os
import torch
import json
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoProcessor, LlavaForConditionalGeneration
from p1_utils import fileNamer
DEBUG_MODE=True

def conver_to_template(conversation):
    """    
    from: conversation={'role': 'user', 'content': [{'type': 'text', 'text': 'Please give me a one sentence caption about the image.'}, {'type': 'image'}]}
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
            cap_output = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            """
            attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when config.output_attentions=True) — 
            Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
            """
            # step: parse output 
            caption=processor.decode(cap_output[0][2:], skip_special_tokens=True)
            caption_dict[os.path.basename(img_p).split('.')[0]]=caption.split(':')[-1]

        # print(caption_dict)
        final_save_path=fileNamer(save_json_path)
        print(f"save at: {final_save_path=}")
        with open(final_save_path, 'w') as f:
            json.dump(caption_dict, f)





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", default="hw3_data/p1_data/images/val", help="Image root")
    parser.add_argument("--save_json_path", default="p1/prediction.json", help="output json file")
    args = parser.parse_args()
    main(data_dir=args.data_root, save_json_path=args.save_json_path)
