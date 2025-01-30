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
from p2_utils import *
from tokenizer import BPETokenizer
from matplotlib import pyplot as plt
DEBUG=False

@torch.no_grad()
def main():
    # args = parser.parse_args()
    # step: IO config
    img_dir='hw3_data/p3_data/images'
    checkpoint="best_ckpt/P2_best.pth"
    images=[os.path.join(img_dir, img) for img in sorted(os.listdir(img_dir))]
    
    # step: training config
    DEVICE= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    LORA_RANK=8

    # step: load model
    model= VitCaptionModel(decoder_ckpt="hw3_data/p2_data/decoder_model.bin", lora_rank=LORA_RANK).to(DEVICE)
    tokenizer=BPETokenizer("encoder.json", "vocab.bpe")
    
    # step: load lora and resize layer
    model.load_state_dict(torch.load(checkpoint), strict=False)
    print(f"using {checkpoint} model")
    
    # step: training settings
    TRANSFORM= timm.data.create_transform(**model.data_config, is_training=False)
    model.eval()
    dict_to_json={}
    for i, img_path in enumerate(images):
        # if i!=1086 and i!=693: continue
        # name= os.path.basename(data["img_path"][0]).split('.')[0]
        image = Image.open(img_path).convert("RGB")
        image =TRANSFORM(image).unsqueeze(0).to(DEVICE)

        # output_ids = model.inference(image, max_length=30) # note : most complete one
        # break
        # output_ids = model.generate(imgs, max_length=30)
        # output_ids = model.beam_search(image, beams=3, max_length=10)
        # output_ids = model.beam_search_ori(image, beams=3, max_length=20)
        output_ids = model.visualize(image, img_path, tokenizer, max_length=20)
        # print(output_folder)
        
        output_ids=trim_tokens(output_ids)
        caption=tokenizer.decode(output_ids)
        name=os.path.basename(img_path).split('.')[0]
        
        dict_to_json[name]=caption

    with open("p3_p2.json", 'w', encoding='utf-8') as f:
        json.dump(dict_to_json, f, ensure_ascii=False, indent=4)
        print(f"prediction caption saved at: p3_p2.json")

        # break
        # imgs, prompt_tokens, gt_tokens, masked_label= data['image'].to(DEVICE), data['prompt'].to(DEVICE), data['gt_label'], data['masked_label'].to(DEVICE)
        # caption_pred=model(imgs, masked_label) 
        # # note : add dummy tensor in order to make the padding size match the prediction size
        # # pbar.set_description(f"pred: {torch.argmax(softmax(caption_pred.swapaxes(1, 2)), axis=-1)[0]}")
        # print(f"pred: {torch.argmax(softmax(caption_pred.swapaxes(1, 2)), axis=-1)}| gt: {gt_tokens[0][:5]}")
                

        
    # ====EOL====

    # with open(json_save_path, 'w', encoding='utf-8') as f:
    #     json.dump(dict_to_json, f, ensure_ascii=False, indent=4)
    #     print(f"prediction caption saved at: {json_save_path}")
if __name__ == '__main__':
    main()
        