#!/bin/bash
python3 p2/inference.py --load_dataset_dir $1 --save_output $2 --load_model "HW1/P2_LRASPP_Mobile_V3.pth"
