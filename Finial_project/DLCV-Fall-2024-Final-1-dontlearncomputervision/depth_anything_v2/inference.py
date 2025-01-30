import os
import cv2
import glob
import clip
import json
import torch
import argparse
import matplotlib
import numpy as np
from PIL import Image
from ultralytics import YOLO
from datasets import load_dataset

from depth_anything_v2.dpt import DepthAnythingV2


# load dataset from the given path
dataset = load_dataset("../data/dlcv_2024_final1", split='test')  # TODO: change dataset path
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)


def generate_depth_map_embeddings():
    """
    output: a list of depth map embeddings
    output embedding list size: 900 * 512, each row is a testing image's depth map embedding
    """
    os.makedirs("./inference_output/depth_map/", exist_ok=True)

    depth_anything = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
    depth_anything.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitb.pth', map_location='cpu')) # TODO: change .pth file's path
    depth_anything = depth_anything.to(DEVICE).eval()
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for i, data in enumerate(dataset):
        # ---------------------------------------------------- generate depth map --------------------------------------------------------------
        # print(f"Generate {i}th testing image's depth map. :)")

        raw_image = data['image']
        # print(type(raw_image)) # PIL.PngImagePlugin.PngImageFile

        # convert input type to numpy.ndarray
        raw_image = np.array(raw_image)

        depth = depth_anything.infer_image(raw_image, 518)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        # print("depth map shape:", depth.shape) # (720, 1355)
        """
        Each entry of depth is a number in [0, 255], which represents the depth information of the corresponding pixel.
        The larger the value, the closer it is to the camera.
        """

        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        # print(type(depth))  # numpy.ndarray
        # print("depth map shape:", depth.shape) # (720, 1355, 3) 
        """
        Each entry of depth is an array of size 3, which represents the color value (BGR) of the corresponding pixel in the output picture.
        """

        cv2.imwrite(os.path.join("./inference_output/depth_map", f'depth_image_{i}.png'), depth)

        # # convert type to PIL
        # depth = depth.astype(np.uint8)
        # pil_image = Image.fromarray(depth)
        # # print(type(pil_image))  # PIL.Image.Image
        # # print(pil_image.size)   # (1355, 720)


        # ---------------------------------------------------- use CLIP to generate embedding of depth map --------------------------------------------------------------
        # image = preprocess(pil_image).unsqueeze(0).to(DEVICE)  # TEST: the input of preprocess should be PIL.PngImagePlugin.PngImageFile, don't know if PIL.Image.Imageis available or not
        # print(type(image))  # torch.Tensor
        # print(image.shape)  # torch.Size([1, 3, 224, 224])

        image = preprocess(Image.open(f"./inference_output/depth_map/depth_image_{i}.png")).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            # print(type(image_features))  # <class 'torch.Tensor'>
            # print(image_features.shape)  # torch.Size([1, 512])

        image_features_2np = image_features.cpu().numpy()  # numpy.ndarray
        # print(image_features_2np, "\n\n\n")

        if i == 0:
            embedding_list = image_features_2np.reshape(1, -1)
        else:
            embedding = image_features_2np.reshape(1, -1)
            embedding_list = np.append(embedding_list, embedding, axis=0)  # numpy.ndarray

    return embedding_list


def generate_original_image_embedding():
    """
    output: a list of original image embeddings
    output embedding list size: 900 * 512, each row is a testing data's embedding
    """

    for i, data in enumerate(dataset):
        # print(f"CLIP: Processing the {i}th original testing image.")

        image = data['image']
        image = preprocess(image).unsqueeze(0).to(DEVICE)
        # print(type(image))  # torch.Tensor
        # print(image.shape)  # torch.Size([1, 3, 224, 224])

        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            # print(type(image_features))  # <class 'torch.Tensor'>
            # print(image_features.shape)  # torch.Size([1, 512])

        image_features_2np = image_features.cpu().numpy()  # numpy.ndarray

        if i == 0:
            embedding_list = image_features_2np.reshape(1, -1)
        else:
            embedding = image_features_2np.reshape(1, -1)
            embedding_list = np.append(embedding_list, embedding, axis=0)  # numpy.ndarray

    return embedding_list


















def generate_segment_image_embedding():
    """
    output: a list of segment image embeddings
    output embedding list size: 900 * 512, each row is a testing data's segment image embedding
    """

    # 載入 YOLO 模型
    yolo_model = YOLO("yolo11l-seg.pt")  # TODO: change .pt file's path

    # 定義輸出路徑
    output_dir = "./inference_output/segment_image/"
    os.makedirs(output_dir, exist_ok=True)

    for i, data in enumerate(dataset):

        # print(f"Generate segmentation of the {i}th testing image. :)")

        # 取得原始圖片
        raw_image = data['image']

        # 執行 YOLO 模型預測
        results = yolo_model.predict(source=raw_image, save=False, save_txt=True, show_boxes=False)  # save=False 表示不儲存到默認路徑

        # 取得繪製後的圖片
        result_image = results[0].plot(boxes=False)  # YOLO 結果支援直接生成繪製後的影像 (NumPy array)

        # 轉換 BGR 到 RGB
        rgb_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

        # 儲存為 .jpg 檔
        save_path = os.path.join(output_dir, f"segment_image_{i}.jpg")
        Image.fromarray(rgb_image).save(save_path)

        # ---------------------------------------------------- use CLIP to generate embedding --------------------------------------------------------------
        image = preprocess(Image.open(f"./inference_output/segment_image/segment_image_{i}.jpg")).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            # print(type(image_features))  # <class 'torch.Tensor'>
            # print(image_features.shape)  # torch.Size([1, 512])

        image_features_2np = image_features.cpu().numpy()  # numpy.ndarray
        # print(image_features_2np, "\n\n\n")

        if i == 0:
            embedding_list = image_features_2np.reshape(1, -1)
        else:
            embedding = image_features_2np.reshape(1, -1)
            embedding_list = np.append(embedding_list, embedding, axis=0)  # numpy.ndarray

    return embedding_list


def generate_bbox_image_embedding():
    """
    output: a list of bbox image embeddings
    output embedding list size: 900 * 512, each row is a testing data's bbox image embedding
    """

    # 載入我們 finetune 過後的 YOLO 模型
    model = YOLO("traffic_sign.pt")  # TODO: change .pt file's path

    # 定義輸出路徑
    output_dir = "./inference_output/bbox_image/"
    os.makedirs(output_dir, exist_ok=True)

    for i, data in enumerate(dataset):

        # print(f"Generate traffic sign bbox of the {i}th test image. :)")

        # 取得原始圖片
        raw_image = data['image']

        # 執行 YOLO 模型預測
        results = model.predict(source=raw_image, save=False, save_txt=True, show_boxes=False)  # save=False 表示不儲存到默認路徑

        # 取得繪製後的圖片
        result_image = results[0].plot(boxes=True, labels=False)  # YOLO 結果支援直接生成繪製後的影像 (NumPy array)

        # 轉換 BGR 到 RGB
        rgb_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

        # 儲存為 i_image.jpg
        save_path = os.path.join(output_dir, f"bbox_image_{i}.jpg")
        Image.fromarray(rgb_image).save(save_path)

        # ---------------------------------------------------- use CLIP to generate embedding --------------------------------------------------------------
        image = preprocess(Image.open(f"./inference_output/bbox_image/bbox_image_{i}.jpg")).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            # print(type(image_features))  # <class 'torch.Tensor'>
            # print(image_features.shape)  # torch.Size([1, 512])

        image_features_2np = image_features.cpu().numpy()  # numpy.ndarray
        # print(image_features_2np, "\n\n\n")

        if i == 0:
            embedding_list = image_features_2np.reshape(1, -1)
        else:
            embedding = image_features_2np.reshape(1, -1)
            embedding_list = np.append(embedding_list, embedding, axis=0)  # numpy.ndarray

    return embedding_list

