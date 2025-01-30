#!/bin/bash
python3 p1/inference.py --load_dataset_csv $1 --load_dataset_root $2 --save_csv_result $3 --load_model "HW1/P1_Resnet50.pth"
# bash hw1_1.sh hw1_data/p1_data/office/val.csv hw1_data/p1_data/office/val p1/test.csv
