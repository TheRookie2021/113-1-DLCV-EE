from transformers import TrainingArguments
from transformers import Trainer

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
import json
import timm
import open_clip
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoTokenizer, AutoImageProcessor
from transformers import TrainingArguments, BitsAndBytesConfig

import loralib as lora
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
"""
split can be: “train”, “val”, “test”
You can use the “streaming” argument to avoid downloading whole data
dataset: 30GB,  https://huggingface.co/datasets/ntudlcv/dlcv_2024_final1
how to use?:    https://huggingface.co/docs/datasets/en/stream 
streaming or local?: https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable
load then preprocess: https://huggingface.co/docs/diffusers/en/training/unconditional_training
"""

def conver_to_template(conversation, ground_truth=None):
    """    
    from: <image>\nThere is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.
    to: prompt= 'USER: <image>\nPlease give me a one sentence caption about the image.\nASSISTANT:'
    
    USER: <image>\n<prompt>\nASSISTANT:
    """    
    # test:
    # template=f'USER: {conversation} ASSISTANT: {ground_truth}'

    # print(conversation)
    template=f'USER: {conversation}\nASSISTANT: {ground_truth}</s>'
    # print(template)
    return template
    
class LLavaDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        image=[]
        text=[]
        # print(f"{len(batch)=}")
        for data in batch:
            # print(data['id'])
            process_text=conver_to_template(data['conversations'][0]['value'], data['conversations'][1]['value'])
            image.append(data['image'])
            text.append(process_text)

        inputs = self.processor(images=image, text=text,  padding=True, return_tensors='pt',) # 1*1211 ok; 1*1443 not ok
        # inputs = self.processor(images=image, text=text,  padding=True, return_tensors='pt',) # 1*1211 ok; 1*1443 not ok
        # print(f"{inputs['pixel_values'].shape=}") # torch.Size([b, 3, 336, 336]) --encoder--> image_features.shape=torch.Size([1, 576, 4096])
        # print(f"{inputs['input_ids'].shape=}") # torch.Size([b, n<2048])
        # print(f"{inputs['input_ids']=}") # torch.Size([b, n<2048])

        EOL_TOKEN=2
        labels=self.processor.tokenizer(data['conversations'][1]['value'])['input_ids']
        # print(f"{labels[0]=}")
        labels=labels[1:]
        labels.append(EOL_TOKEN)
        labels=torch.tensor(labels).unsqueeze(0)
        
        # step: pad ground_truth
        paddings= max(inputs['input_ids'].shape[1], labels.shape[1]) - labels.shape[1]
        m = torch.nn.ConstantPad2d((paddings, 0), -100)
        labels=m(labels)

    
        # TODO: mask label
        # labels = inputs["input_ids"].clone()
        # print(labels)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        # print(f"{labels.shape=}") # torch.Size([b, n<2048])
        # print(labels)
        inputs['labels']=labels
        # print(inputs)
        return inputs


def train(args):

    """
    Fine-tune a pretrained model: https://huggingface.co/docs/transformers/en/training
    example1: https://colab.research.google.com/github/mrm8488/shared_colab_notebooks/blob/master/fine_tune_VLM_LlaVa.ipynb#scrollTo=ee1IMXszqWyU
    
    """
    # EPOCH=1
    # step: load and prepare model, https://huggingface.co/llava-hf/llava-1.5-7b-hf
    # model_id = "llava-hf/llava-1.5-7b-hf"
    model_id = "finetune_llava-1.5-7b-hf_lora_5_weight_decay/checkpoint-57620"
    bnb_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        quantization_config=bnb_config
        # torch_dtype=torch.float16, 
        # low_cpu_mem_usage=True,
        # load_in_4bit=True
    )

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    data_collator = LLavaDataCollator(processor)

    # print(processor)

    peft_config = LoraConfig(
        target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj"],
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=16, lora_alpha=256, lora_dropout=0.2)
    model = get_peft_model(model, peft_config) 
    # # ====freeze vision encoder====
    # for param in model.vision_tower.parameters():
    #     param.requires_grad = False
    model.print_trainable_parameters()

    # step: load and prepare dataset
    dataset= load_dataset("ntudlcv/dlcv_2024_final1", split='train')
    dataset_test= load_dataset("ntudlcv/dlcv_2024_final1",split='val')

    dataset= dataset.with_format("torch")
    dataset_test= dataset_test.with_format("torch")
    print(len(dataset))
    # print(dataset[0])
    # print(dataset_test[0])
    # # step: Training setting
    training_args = TrainingArguments(
        output_dir="finetune_llava-1.5-7b-hf_lora_5_weight_decay",
        num_train_epochs=3,
        # max_steps=,
        do_train=True,
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        lr_scheduler_type ="cosine",
        weight_decay=0.001,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,

        remove_unused_columns=False,
        dataloader_pin_memory=False,
        push_to_hub=False,
    )

    # step: Train and save
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset_test,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=True)
    model.print_trainable_parameters()
    trainer.save_model('model_output')

    
if __name__=="__main__":
    args=None
    train(args)
    