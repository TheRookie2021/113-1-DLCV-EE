# DLCV: Don't Learn Computer Vision

## Environment Setup
Follow these steps to set up your environment:

1. Setup LLaVA
```bash
$ git clone https://github.com/haotian-liu/LLaVA.git
$ cd LLaVA

$ conda create -n llava python=3.10 -y
$ conda activate llava
$ pip install --upgrade pip  # enable PEP 660 support
$ pip install -e .

$ cd ..
```
2. Setup Depth-Anything-v2
```bash
# Please ensure you are in the llava env

$ pip install gradio_imageslider
$ pip install gradio==4.29.0
$ pip install matplotlib
$ pip install opencv-python
$ pip install torch
$ pip install torchvision
$ pip install datasets
$ pip install -r requirements.txt
```
3. Setup YOLOv11
```bash
# Please ensure you are in the llava env

$ pip install ultralytics
```
4. Setup CLIP
```bash
# Please ensure you are in the llava env

$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```
5. Setup FAISS
```bash
# Please ensure you are in the llava env

$ pip install faiss-gpu
```

6. Install gdown
```bash
# Please ensure you are in the llava env

$ pip install gdown
```

## Model Checkpoint Download
```bash
# Please ensure you are in the llava env

$ python download.py
```


## Inference (with RAG)
```bash
# Please ensure you are in the llava env
# Run this command to reproduce our result 
# Can modify json filename ang batch size in the shell script

$ bash inference_RAG.sh
```

## Inference (with Lora only)
```bash
# Please ensure you are in the llava env
# Run this command to reproduce our result 
# Can modify json filename ang batch size in the shell script
# best score (the one on the leaderboard)

$ bash inference_finetuned_lora.sh
```