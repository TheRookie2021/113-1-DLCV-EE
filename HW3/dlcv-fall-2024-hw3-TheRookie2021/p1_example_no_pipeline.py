import requests
from PIL import Image
import os
import torch
import json
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
from p1_utils import fileNamer
# step: input, output files
DEBUG_MODE=True
data_dir="hw3_data/p1_data/images/val"
save_json_name="p1_captions.json"
caption_dict={}
# image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
filepaths=[os.path.join(data_dir, name) for name in sorted(os.listdir(data_dir)) ]
model_id = "llava-hf/llava-1.5-7b-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    load_in_4bit=True
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

# Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image") 
conversations = [
    # CIDEr: 1.124212293293333 | CLIPScore: 0.782391357421875 
    # {
    #   "role": "user",
    #   "content": [
    #       {"type": "text", "text": "Describe the image in one sentence caption."},
    #       {"type": "image"},
    #     ],
    # },
    # {
    #   "role": "user",
    #   "content": [
    #       {"type": "text", "text": "What do you see in the given image? Describe the image in one sentence."},
    #       {"type": "image"},
    #     ],
    # },
    # {
    #   "role": "user",
    #   "content": [
    #       {"type": "text", "text": "Please tell me about the image in one sentence."},
    #       {"type": "image"},
    #     ],
    # },    
    # best: CIDEr: 1.1604811812238933 | CLIPScore: 0.77781494140625
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "Please give me a one sentence caption about the image."},
          {"type": "image"},
        ],
    },    
    # {
    #   "role": "user",
    #   "content": [
    #       {"type": "text", "text": "Caption the image in one sentence."},
    #       {"type": "image"},
    #     ],
    # },
]


for conversation in conversations:
    prompt = processor.apply_chat_template([conversation], add_generation_prompt=True)
    if DEBUG_MODE: print("======== start inferencing ========")
    if DEBUG_MODE: print(f"{prompt=}")
    
    for img_p in tqdm(filepaths, total=len(filepaths)):
        # print(f"======== {img_p} ========")
        # raw_image = Image.open(requests.get(image_file, stream=True).raw)
        image = Image.open(img_p)
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)

        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        caption=processor.decode(output[0][2:], skip_special_tokens=True)
        
        # step: parse output 
        # print(f"{os.path.basename(img_p).split('.')[0]}, {caption.split(':')[-1]}")
        caption_dict[os.path.basename(img_p).split('.')[0]]=caption.split(':')[-1]

    # print(caption_dict)
    with open(fileNamer('p1_result.json'), 'w') as f:
        json.dump(caption_dict, f)
