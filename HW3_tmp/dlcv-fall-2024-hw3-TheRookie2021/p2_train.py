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
from argparse import ArgumentParser
DEBUG=False
if __name__ == '__main__':
    parser = ArgumentParser()
    # TODO: wire the 
    parser.add_argument("--train_proj_only", action='store_true', help="Prediction json file")
    parser.add_argument("--finetune_on", default=None, type=str, help="ckpt to be finetuned")
    parser.add_argument("--lr", default=0.0002, type=float, help="ckpt to be finetuned")
    parser.add_argument("--lora", default=16, type=int, help="ckpt to be finetuned")

    parser.add_argument("--images_root", default="hw3_data/p2_data/images", help="Image root")
    parser.add_argument("--annotation_file", default="p3_data/val.json", help="Annotation json file")
    
    parser.add_argument("--checkpoint_save_folder", default="p2_models/Round31_find_dataset_bug", help="Annotation json file")
    parser.add_argument("--proj_save_path", default="p2_models/Round4_proj_only/img_proj.pt", help="Annotation json file")

    args = parser.parse_args()

    # step: IO config
    img_dir='hw3_data/p2_data/images'
    annotation_path='hw3_data/p2_data'
    checkpoint_save_folder=args.checkpoint_save_folder
    proj_save_path=args.proj_save_path
    to_be_finetuned=args.finetune_on
    # to_be_finetuned="p2_models/Round24_resume_round23_lr1-e4/ep3_loss2.9090226618245767.pth"
    os.makedirs(checkpoint_save_folder,exist_ok=True)
    os.makedirs(os.path.dirname(proj_save_path),exist_ok=True)

    # step: training config
    print(f"{args.train_proj_only=}")
    TRAIN_PROJ_ONLY=args.train_proj_only
    FINETUNE_MODE= False if to_be_finetuned == None else True
    BATCH_SIZE= 32
    EPOCH= 10
    LR=args.lr
    print(f"{LR=}")
    DEVICE= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    LORA_RANK=0 if TRAIN_PROJ_ONLY else args.lora
    print(f"{TRAIN_PROJ_ONLY=}, {LORA_RANK=}")
    MAX_LENGTH=30 # TODO : variable need to be deleted
    PAD_FOR_GT=-1 

    # step: load model
    model= VitCaptionModel(decoder_ckpt="hw3_data/p2_data/decoder_model.bin", lora_rank=LORA_RANK).to(DEVICE)
    tokenizer=BPETokenizer("encoder.json", "vocab.bpe")
    
    # step: load data
    TRANSFORM= timm.data.create_transform(**model.data_config, is_training=False)
    train_dataset=  CustomImageDataset(img_dir=os.path.join(img_dir, "train"), 
                                    annotaion_path=os.path.join(annotation_path,'train.json'), 
                                    transform=TRANSFORM,
                                    max_len=MAX_LENGTH,
                                    tokenizer=tokenizer)
    test_dataset=   CustomImageDataset(img_dir=os.path.join(img_dir, "val"), 
                                    annotaion_path=os.path.join(annotation_path, 'val.json'), 
                                    transform=TRANSFORM,
                                    tokenizer=tokenizer)
    train_dataloader=   DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=train_dataset.collate_fn_padd, shuffle=True)
    test_dataloader =   DataLoader(test_dataset, batch_size=int(BATCH_SIZE/4), collate_fn=test_dataset.collate_fn_padd, shuffle=True)

    # step: training mode: [proj layer only, lora+ proj layer, finetune all]
    lora.mark_only_lora_as_trainable(model)
    print(f"======== Trainable model param on {LORA_RANK=}: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M/{sum(p.numel() for p in model.parameters())/ 1e6}M ========")

    if FINETUNE_MODE:
        model.load_state_dict(torch.load(to_be_finetuned),strict=False)
        print(f"finetune mode(lora+ proj layers): using {to_be_finetuned=}")
        for param in model.img_projection_layer.parameters():
            param.requires_grad = True

        save_loss_plot=checkpoint_save_folder
    elif TRAIN_PROJ_ONLY:
        print(f"pretrain on proj layer...")
        for param in model.img_projection_layer.parameters():
            param.requires_grad = True
        
        save_loss_plot=checkpoint_save_folder
    else: 
        model.load_state_dict(torch.load(proj_save_path),strict=False)
        print(f"lora mode(train new lora layers), freeze pretrained proj layer: using {proj_save_path=}")
        for param in model.img_projection_layer.parameters():
            param.requires_grad = True
        save_loss_plot=checkpoint_save_folder
        
        

    print(f"======== train img_projection_layer : {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M/{sum(p.numel() for p in model.parameters())/ 1e6}M ========")

    # step: training settings
    lora_para=[p for name, p in model.named_parameters() if 'lora' in name]
    proj_para= model.img_projection_layer.parameters()
    optimizer = torch.optim.Adam( [{'params': proj_para, 'lr': LR},
                                   {'params': lora_para, 'lr': LR}], 
                                   lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    criterion =CrossEntropyLoss(ignore_index=PAD_FOR_GT) 
    trainable_weights = [ name for name, param in model.named_parameters() if param.requires_grad == True]
    
    
    if DEBUG: print(trainable_weights)
    softmax=nn.Softmax(dim=-1)
    
    # step: start training loop
    best_loss=float('inf')
    plot_loss=[]
    plot_evl_loss=[]
    
    try:
        pbar=tqdm(range(1))
        for ep in tqdm(range(EPOCH)):
            # if DEBUG: break
            model.train()
            loss_record=[]
            for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
                if DEBUG: print(imgs, prompt_tokens, gt_tokens)
                optimizer.zero_grad()
                imgs, prompt_tokens, gt_tokens, masked_label= data['image'].to(DEVICE), data['prompt'].to(DEVICE), data['gt_label'], data['masked_label'].to(DEVICE)
                caption_pred=model(imgs, masked_label) 
                caption_pred = torch.swapaxes(caption_pred, 1, 2) #note: put it to the outmost layer

                if DEBUG: print("caption_pred | gt_tokens")
                if DEBUG: print(caption_pred.shape, gt_tokens.shape)
                # break
                # note : add dummy tensor in order to make the padding size match the prediction size
                output_seq_len=caption_pred.shape[-1]
                gt_tokens.append(torch.tensor([0]*output_seq_len)) # add dummy tensor to match the output of the decoder
                gt_tokens= torch.nn.utils.rnn.pad_sequence(gt_tokens, padding_value=PAD_FOR_GT).swapaxes(0,1)
                # print(gt_tokens)
                # print(masked_label)
                # print(caption_pred.shape, gt_tokens[:-1].shape)
                # pbar.set_description(f"pred: {torch.argmax(softmax(caption_pred.swapaxes(1, 2)), axis=-1)[0]}")
                # print(f"pred: {torch.argmax(softmax(caption_pred.swapaxes(1, 2)), axis=-1)[0][:5]}\t| gt: {gt_tokens[0][:5]}")
                # print(f"pred: {torch.argmax(softmax(caption_pred.swapaxes(1, 2)), axis=-1)[0][-5:]}\t| gt: {gt_tokens[0][-5:]}\n")
                loss= criterion(caption_pred, gt_tokens[:-1].to(DEVICE))
                loss.backward()
                optimizer.step()
                loss_record.append(loss.cpu().detach().numpy())
            
            # step: evaluation part
            model.eval()
            eval_loss=[]
            for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                imgs, prompt_tokens, gt_tokens, masked_label= data['image'].to(DEVICE), data['prompt'].to(DEVICE), data['gt_label'], data['masked_label'].to(DEVICE)
                caption_pred=model(imgs, masked_label) # TODO: should it be prompt or masked gt caption? => it's okay to use captions
                caption_pred = torch.swapaxes(caption_pred, 1, 2) #note: put it to the outmost layer

                # note : add dummy tensor inorder to make the padding size match the prediction size
                output_seq_len=caption_pred.shape[-1]
                gt_tokens.append(torch.tensor([0]*output_seq_len)) # add dummy tensor to match the output of the decoder
                gt_tokens= torch.nn.utils.rnn.pad_sequence(gt_tokens, padding_value=PAD_FOR_GT).swapaxes(0,1)
                
                loss= criterion(caption_pred, gt_tokens[:-1].to(DEVICE))
                eval_loss.append(loss.cpu().detach().numpy())
            
            # step: lr scheduler update
            scheduler.step()
            print(f"{ep=}, lr={float(scheduler.get_last_lr()[0])}")
            
            # step: analysis result of the epoch
            avg_loss=(sum(loss_record)/len(loss_record))
            avg_eval_loss=(sum(eval_loss)/len(eval_loss))
            plot_loss.append(avg_loss)
            plot_evl_loss.append(avg_eval_loss)
            print(f"==== {ep=}/{EPOCH}: train loss: {avg_loss}| eval loss: {avg_eval_loss} ==== ")
            
            # step: saving the lora para and the embeding resize layer
            if best_loss>avg_loss:
                best_loss= avg_loss
                save_path=os.path.join(checkpoint_save_folder, f"ep{ep}_loss{avg_loss}.pth") if not TRAIN_PROJ_ONLY else proj_save_path
                print(f"save model at {save_path}")
                save_weights = {k: v for k, v in model.state_dict().items() if k in trainable_weights }
                torch.save(save_weights, save_path) # note: only save trainable weight 
        plot_loss_graph(save_loss_plot, plot_loss, plot_evl_loss, "train_loss", "eval_loss" )
    except KeyboardInterrupt:  
        print("=== interrupt ===")  
        plot_loss_graph(save_loss_plot, plot_loss, plot_evl_loss, "train_loss", "eval_loss" )