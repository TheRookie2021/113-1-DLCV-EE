#!/bin/bash
# TODO - run your inference Python3 code
python3 p2_inference.py --model best_ckpt/P2_best.pth --folder $1 --save_json_path $2  --decoder $3
# python3 p2_inference.py --model p2_models/Round34_set_alpha16_add_dropout_to_other_lora_layers/ep3_loss1.977310167020544.pth
# $1: path to the folder containing test images (e.g. hw3/p2_data/images/test/)
# $2: path to the output json file (e.g. hw3/output_p2/pred.json) 
# $3: path to the decoder weights (e.g. hw3/p2_data/decoder_model.bin) (This means that you don’t need to upload decoder_model.bin)
