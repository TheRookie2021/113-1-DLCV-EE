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
DEBUG=False

if __name__ == '__main__':
    # args = parser.parse_args()
    # step: IO config
    img_dir='hw3_data/p2_data/images'
    annotation_path='hw3_data/p2_data'
    checkpoint_folder="p2_models/Round3"
    os.makedirs(checkpoint_folder,exist_ok=True)

    # step: training config
    BATCH_SIZE= 64
    EPOCH= 20
    DEVICE= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    LORA_RANK=32
    MAX_LENGTH=80 

    # step: load model
    model= ClipCaptionModel(decoder_ckpt="hw3_data/p2_data/decoder_model.bin", max_len=MAX_LENGTH, lora_rank=LORA_RANK).to(DEVICE)
    tokenizer=BPETokenizer("encoder.json", "vocab.bpe")
    
    # step: decide para to be train
    lora.mark_only_lora_as_trainable(model)
    for param in model.feature_projection.parameters():
        param.requires_grad = True

    print(f"======== Trainable model param on {LORA_RANK=}: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M/{sum(p.numel() for p in model.parameters())/ 1e6}M ========")

    # step: load data
    train_dataset=  CustomImageDataset(img_dir=os.path.join(img_dir, "train"), 
                                    annotaion_path=os.path.join(annotation_path,'train.json'), 
                                    transform=model.preprocess,
                                    max_len=MAX_LENGTH,
                                    tokenizer=tokenizer)
    test_dataset=   CustomImageDataset(img_dir=os.path.join(img_dir, "val"), 
                                    annotaion_path=os.path.join(annotation_path, 'val.json'), 
                                    transform=model.preprocess,
                                    tokenizer=tokenizer)
    train_dataloader=   DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader =   DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # step: training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterion =CrossEntropyLoss(ignore_index=-100)
    trainable_weights = [ name for name, param in model.named_parameters() if param.requires_grad == True]
    if DEBUG: print(trainable_weights)

    # step: start training loop
    best_loss=float('inf')
    plot_loss=[]
    for ep in tqdm(range(EPOCH)):
        # if DEBUG: break
        model.train()
        loss_record=[]
        for i, (imgs, prompt_tokens, gt_tokens, img_path, _) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # if DEBUG: print(imgs, prompt_tokens, gt_tokens)
            optimizer.zero_grad()
            imgs, prompt_tokens, gt_tokens= imgs.to(DEVICE), prompt_tokens.to(DEVICE), gt_tokens.to(DEVICE)
            
            caption_pred=model(imgs, prompt_tokens)
            if DEBUG: print("caption_pred | gt_tokens")
            if DEBUG: print(caption_pred.shape, gt_tokens.shape)
            if DEBUG: print(caption_pred.dtype, gt_tokens.dtype)
            # cap_tok=tokenizer.decode(caption_pred)
            
            loss= criterion(caption_pred, gt_tokens)
            
            loss_record.append(loss)

            loss.backward()
            optimizer.step()
        avg_loss=(sum(loss_record)/len(loss_record))
        plot_loss.append(avg_loss)
        print(f"{ep=}/{EPOCH}: {avg_loss=}")
        break
        # step: saving the lora para and the embeding resize layer
        if best_loss>avg_loss:
            best_loss= avg_loss
            save_path=os.path.join(checkpoint_folder, f"ep{ep}_loss{avg_loss}.pth")
            print(f"save model at {save_path}")
            save_weights = {k: v for k, v in model.state_dict().items() if k in trainable_weights }
            torch.save(save_weights, save_path) # note: only save trainable weight 
           
        # # step: evaluate
        # model.eval()
    save_path=os.path.join(checkpoint_folder, f"train_loss_plot.png")
    print(f"loss fig: {save_path}")
    plt.plot(range(len(plot_loss.cpu().numpy())), plot_loss)  # Plot the chart
    plt.savefig(save_path)  
    
        
# TODO: 
# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning?tab=readme-ov-file#implementation
# https://github.com/inuwamobarak/Image-captioning-ViT/blob/main/fintuning/fintuning_for_image_caption_on_custom_dataset.ipynb