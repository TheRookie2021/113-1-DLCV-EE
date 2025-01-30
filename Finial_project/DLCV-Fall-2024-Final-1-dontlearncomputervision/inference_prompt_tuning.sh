# python inference/inference_only.py --use_prompt_tuning --model llava-hf/llava-1.5-7b-hf --save_json_path inference/prompt_tuning.json --batch_size 16
python inference/inference.py --use_prompt_tuning --model finetune_llava-1.5-7b-hf_lora_5_weight_decay/checkpoint-86430 --save_json_path inference/submission_10_lora_r16_ep3_beam3.json --batch_size 4
