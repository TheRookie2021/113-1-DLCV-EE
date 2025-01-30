#!/bin/bash

# TODO - run your inference Python3 code
python3 p1_caption.py --data_root $1 --save_json_path $2
# $1: path to the folder containing test images (e.g. hw3/p1_data/images/test/)
# $2: path to the output json file (e.g. hw3/output_p1/pred.json)
# bash hw3_1.sh hw3_data/p1_data/images/val p1/prediction.json