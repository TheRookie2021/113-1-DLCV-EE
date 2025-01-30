"""
- Use pretrained ***visual encoder*** and ***text decoder*** for image captioning.
    - Input: RGB image
    - Output: image caption (text)
- You are limited to use ***transformer encoder + decoder architecture*** in this assignment.
    - Language model: pretrained transformer-base decoder
- Implement PEFT (Parameter-Efficient Fine-Tuning) for the task of image captioning (from scratch)
    - Lora

# ref: 
    - https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/gpt2_ft.py
    - https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/model.py

"""
import torch
from torch.utils.data import DataLoader
import torchvision.transforms  as transforms
import loralib as lora
from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss
from p2_utils import *
from tokenizer import BPETokenizer
from matplotlib import pyplot as plt

from argparse import ArgumentParser
DEBUG=False

@torch.no_grad()
def main(args):
    # step: IO config
    img_dir=args.folder
    json_save_path=args.save_json_path
    decoder_path=args.decoder
    checkpoint=args.model
    images=[os.path.join(img_dir, img) for img in sorted(os.listdir(img_dir))]
    os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
    print(len(images))
    # step: training config
    DEVICE= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    LORA_RANK=8

    # step: load model
    model= VitCaptionModel(decoder_ckpt=decoder_path, lora_rank=LORA_RANK).to(DEVICE)
    tokenizer=BPETokenizer("encoder.json", "vocab.bpe")
    
    # step: load lora and resize layer
    model.load_state_dict(torch.load(checkpoint), strict=False)
    print(f"using {checkpoint} model")
        
    # step: training settings
    TRANSFORM= timm.data.create_transform(**model.data_config, is_training=False)
    model.eval()
    dict_to_json={}
    max_len=0
    for i, img_path in tqdm(enumerate(images), total=len(images)):
        # name= os.path.basename(data["img_path"][0]).split('.')[0]
        image = Image.open(img_path).convert("RGB")
        image =TRANSFORM(image).unsqueeze(0).to(DEVICE)

        output_ids = model.inference(image, max_length=50)
        # output_ids = model.beam_search_ori(image, beams=3, max_length=50)
        max_len=max(max_len, len(trim_tokens(output_ids)))
        # output_ids = model.inference(image, max_length=50)
        sentence = tokenizer.decode(trim_tokens(output_ids))
        name=os.path.basename(img_path).split('.')[0]
        dict_to_json[name]=sentence
        print(f"{name}: {sentence}")
        
    # ====EO-LOOP====

    with open(json_save_path, 'w', encoding='utf-8') as f:
        json.dump(dict_to_json, f, ensure_ascii=False, indent=4)
        print(f"prediction caption saved at: {json_save_path}")
        print(f"{max_len=}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--folder", default="hw3_data/p2_data/images/val", help="Image root")
    parser.add_argument("--save_json_path", default="p2_prediction.json", help="output json file")
    parser.add_argument("--decoder", default="hw3_data/p2_data/decoder_model.bin", help="output json file")
    parser.add_argument("--model", default=None, help="Prediction json file")

    args = parser.parse_args()
    
    main(args)
        