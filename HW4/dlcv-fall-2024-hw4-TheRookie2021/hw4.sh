#!/bin/bash
python3 gaussian-splatting/render.py -m HW4/round3 -s $1 --save_path $2
# TODO - run your inference Python3 code
# $1: path to the folder of split (e.g., */dataset/private_test)
# It also contains the folder of sparse/0/. The camera poses are in sparse/0/cameras.txt and sparse/0/images.txt.
# You should predict novel views base on the private test split.
# $2: path of the folder to put output images (e.g., xxxxxxxxx.png, please follows sparse/0/images.txt to name
# your output images.)
# The filename should be {id}.png (e.g. xxxxxxxxx.png). The image size should be the same as training set.