from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import os
import json
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import LlavaForConditionalGeneration, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from utils import *
"""
split can be: “train”, “val”, “test”
You can use the “streaming” argument to avoid downloading whole data
dataset: 30GB,  https://huggingface.co/datasets/ntudlcv/dlcv_2024_final1
how to use?:    https://huggingface.co/docs/datasets/en/stream 
streaming or local?: https://huggingface.co/docs/datasets/en/about_mapstyle_vs_iterable
load then preprocess: https://huggingface.co/docs/diffusers/en/training/unconditional_training
"""

def inference(args):

    """
    Fine-tune a pretrained model: https://huggingface.co/docs/transformers/en/training
    example1: https://colab.research.google.com/github/mrm8488/shared_colab_notebooks/blob/master/fine_tune_VLM_LlaVa.ipynb#scrollTo=ee1IMXszqWyU
    """

    model_id = args.model
    save_json_path=args.save_json_path
    print("using ", model_id)
    print("save at ", save_json_path)

    # step: load and prepare model, https://huggingface.co/llava-hf/llava-1.5-7b-hf
    bnb_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        quantization_config=bnb_config
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id, 
    #     device_map="auto",
    #     torch_dtype="auto"
    # )
    model.eval()

    # step: load and prepare dataset
    dataset_test= load_dataset("ntudlcv/dlcv_2024_final1",split='test')
    dataset_test= dataset_test.with_format("torch")
    print("--------------------------------------------------------------------------")
    print(f"==== {len(dataset_test)=} ====")
    print(f"==== {args.use_prompt_tuning=} ====")
    print(f"==== {args.use_RAG=} ====")
    task_names=['general', 'regional', 'suggestion']
    # TODO: write the correct path for vector_db_path and metadata_path
    vector_db_path= {'general':"./inference/RAG_dataset/original_depth_yolo_train_general.index", 
                     'regional':"./inference/RAG_dataset/original_depth_yolo_train_regional.index",
                     'suggestion': "./inference/RAG_dataset/original_depth_yolo_train_suggestion.index"}
    
    metadata_path=  {'general':"./inference/metadata/metadata_train_general.json", 
                     'regional':"./inference/metadata/metadata_train_regional.json",
                     'suggestion': "./inference/metadata/metadata_train_suggestion.json" }
            
    preprocess_class=DataPreprocess(use_RAG=args.use_RAG, llava_model=model, model_id="llava-hf/llava-1.5-7b-hf",vector_db_path=vector_db_path, metadata_path=metadata_path )
    
    processor = preprocess_class.processor
    preprocess=preprocess_class.preprocess_data
    if args.use_RAG: # highest priorority
        print("==== use_RAG ====")
        preprocess= preprocess_class.preprocess_data_RAG_prompt_tuning
    elif args.use_prompt_tuning:
        print("==== use_prompt_tuning ====")
        preprocess= preprocess_class.preprocess_data_prompt_tuning
    else:
        print("==== use default ====")
        
    # step: seperate dataset into three different datasets for inference using "filter", https://huggingface.co/docs/datasets/en/process#shuffle
    dataset_test=[dataset_test.filter(function=lambda example: task_name in example["id"], batch_size=16) for task_name in task_names]
    for task in dataset_test: print(len(task))
    dataloader_test=[ DataLoader(task, batch_size=args.batch_size, collate_fn=preprocess, shuffle=False) for task in dataset_test]
    print(f"====finish data loading====")
    
    # step: IO setting 
    caption_dict={}
    if os.path.exists(save_json_path):
        with open(save_json_path, "r") as f:  # reading a file
            caption_dict = json.load(f)  # deserialization

    # step: inference
    for i, task_i_dataloader in enumerate(dataloader_test):
        print(f"==== inference task: {task_names[i]} ====")
        for _, (inputs, image_names) in tqdm(enumerate(task_i_dataloader), total=len(task_i_dataloader)):
            # print(inputs)
            if image_names[0] in caption_dict: continue #for resume
            
            # step: model 
            inputs=inputs.to(0)
            cap_output = model.generate(**inputs, max_new_tokens=500, num_beams=1, do_sample=False)

            # step: saving
            for j, image_name in enumerate(image_names):
                caption=processor.decode(cap_output[j][2:], skip_special_tokens=True) # step: decode and parse output 
                image_name=image_names[j]
                content=caption.split(':')[-1]
                caption_dict[image_name]=content
                with open(save_json_path, 'w') as f:
                    json.dump(caption_dict, f, indent=2)
        print(f"save at: {save_json_path=}")

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf", help="Image root")
    parser.add_argument("--save_json_path", default="./output/baseline.json", help="output json file")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--use_prompt_tuning", action='store_true', help="use_prompt_tuning or not")
    parser.add_argument("--use_RAG", action='store_true', help="use_RAG for prompt tuning or not")
    args = parser.parse_args()
    
    inference(args)
    