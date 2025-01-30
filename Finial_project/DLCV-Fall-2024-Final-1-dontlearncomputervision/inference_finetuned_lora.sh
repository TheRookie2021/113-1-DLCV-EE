# python inference/inference.py --model finetune_llava-1.5-7b-hf_lora_6_weight_decay_continue/checkpoint-7200 --save_json_path inference/submission_7_finetuned_lora_5_resume_r16_maxT300.json --batch_size 16
# python inference/inference.py --model finetune_llava-1.5-7b-hf_lora_3_2/checkpoint-28810 --save_json_path inference/submission_3_finetuned_lora_3_2_no_shuffle.json --batch_size 16 && /
# python inference/inference.py --model finetune_llava-1.5-7b-hf_lora_5_weight_decay/checkpoint-86430 --save_json_path inference/submission_10_lora_r16_ep3_beam3.json --batch_size 4
python inference/inference.py --model finetune_llava-1.5-7b-hf_lora_5_weight_decay/checkpoint-86430 --save_json_path inference/submission.json --batch_size 4
