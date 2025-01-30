import os
import cv2
import clip
import glob
import json
import faiss
import torch
import argparse
import matplotlib
import numpy as np
from PIL import Image
from ultralytics import YOLO
from datasets import load_dataset
import torchvision
from transformers import AutoProcessor, AutoTokenizer
from depth_anything_v2.dpt import DepthAnythingV2

# TODO: change the path of each model's checkpoint
depth_anything_ckpt_path = 'inference/model_checkpoints/depth_anything_v2_vitb.pth'
yolov11_ckpt_path = "inference/model_checkpoints/yolo11l-seg.pt"
finetuned_yolov11_ckpt_path = 'inference/model_checkpoints/traffic_sign.pt'

# hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)

# depth_anything model
depth_anything = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768])
depth_anything.load_state_dict(torch.load(depth_anything_ckpt_path, map_location='cpu'))
depth_anything = depth_anything.to(DEVICE).eval()

# yolov11 model
yolo_model = YOLO(yolov11_ckpt_path)

# finetuned yolov11 model
yolo_finetuned_model = YOLO(finetuned_yolov11_ckpt_path)  # 載入我們 finetune 過後的 YOLO 模型


def generate_original_image_embedding(input_img):
    """
    input_img: data['image']
    output: input image's embedding (size: 1 * 512)
    """
    image = preprocess(input_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    original_embedding = image_features.cpu().numpy()  # numpy.ndarray
    return original_embedding


def generate_depth_map_embedding(input_img):
    """
    input_img: data['image']
    output: input image's depth map embedding (size: 1 * 512)
    """
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    raw_image = np.array(input_img)
    depth = depth_anything.infer_image(raw_image, 518)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    # # convert type to PIL
    # depth = depth.astype(np.uint8)
    # pil_image = Image.fromarray(depth)
    # # print(type(pil_image))  # PIL.Image.Image
    # # print(pil_image.size)   # (1355, 720)
    # image = preprocess(pil_image).unsqueeze(0).to(DEVICE)  # TEST: the input of preprocess should be PIL.PngImagePlugin.PngImageFile, don't know if PIL.Image.Imageis available or not
    # # print(type(image))  # torch.Tensor
    # # print(image.shape)  # torch.Size([1, 3, 224, 224])

    os.makedirs("./inference_output/depth_map/", exist_ok=True)
    cv2.imwrite(os.path.join("./inference_output/depth_map", 'depth_image.png'), depth)
    image = preprocess(Image.open("./inference_output/depth_map/depth_image.png")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    depth_embedding = image_features.cpu().numpy()  # numpy.ndarray
    return depth_embedding


def generate_segment_image_embedding(input_img):
    """
    input_img: data['image']
    output: input image's segment image embedding (size: 1 * 512)
    """
    results = yolo_model.predict(source=input_img, save=False, save_txt=True, show_boxes=False)  # save=False 表示不儲存到默認路徑
    result_image = results[0].plot(boxes=False)  # YOLO 結果支援直接生成繪製後的影像 (NumPy array)
    rgb_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    output_dir = "./inference_output/segment_image/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "segment_image.jpg")
    Image.fromarray(rgb_image).save(save_path)

    image = preprocess(Image.open("./inference_output/segment_image/segment_image.jpg")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    segment_embedding = image_features.cpu().numpy()  # numpy.ndarray

    return segment_embedding


def generate_bbox_image_embedding(input_img):
    """
    input_img: data['image']
    output: input image's bbox image embedding (size: 1 * 512)
    """
    results = yolo_finetuned_model.predict(source=input_img, save=False, save_txt=True, show_boxes=False)  # save=False 表示不儲存到默認路徑
    result_image = results[0].plot(boxes=True, labels=False)  # YOLO 結果支援直接生成繪製後的影像 (NumPy array)
    rgb_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

    output_dir = "./inference_output/bbox_image/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "bbox_image.jpg")
    Image.fromarray(rgb_image).save(save_path)

    image = preprocess(Image.open("./inference_output/bbox_image/bbox_image.jpg")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    bbox_embedding = image_features.cpu().numpy()  # numpy.ndarray

    return bbox_embedding


# TODO: embed a pil image into a concated embedding using other model(sam, clip, ...) 
def embed_fn(input_image):
    """
    input: a pil image
    output: a embedding
    """
    original_embedding = generate_original_image_embedding(input_image)
    depth_map_embedding = generate_depth_map_embedding(input_image)
    segment_image_embedding = generate_segment_image_embedding(input_image)
    # bbox_image_embedding = generate_bbox_image_embedding(input_image)

    # concated_embedding = np.concatenate((original_embedding, depth_map_embedding, segment_image_embedding, bbox_image_embedding), axis=1)
    concated_embedding = np.concatenate((original_embedding, depth_map_embedding, segment_image_embedding), axis=1)
    # print(f"concated_embedding shape: {concated_embedding.shape}")  # (1, 2048)

    return concated_embedding


def retrieve_similar_images(query_embedding, index, top_k=3):
    """
    input: one query_embedding (assume it is a tensor on gpu)
    output: 
        - distances: similarity of images 
        - indices: return the top-k indice of the 2D array, n*k, n= number of query_embeddings, k= top_k values
    """
    query_embedding=query_embedding.astype(np.float32)
    # query_embedding=query_embedding.detach().numpy().astype(np.float32)
    # query_vectors = np.array([query_embedding])
    
    distances, indices = index.search(query_embedding, top_k)
    # print(indices)
    # retrieved_images = [image_paths[int(idx)] for idx in indices[0]]
    return distances[0], indices[0] 


class DataPreprocess():
    def __init__(self, use_RAG=False, llava_model=None, model_id="llava-hf/llava-1.5-7b-hf", vector_db_path=None, metadata_path=None):

        self.task_names=['general', 'regional', 'suggestion']
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id) # Add to test text to text generation

        self.use_RAG=use_RAG
        self.vector_db_path=vector_db_path
        self.metadata_path=metadata_path
        
        self.RAG_index={}
        self.metadata={}

        self.llava_model = llava_model

        # TODO: adding RAG into data processing 
        if self.use_RAG: 
            for task_name in self.task_names:
                assert task_name in vector_db_path and task_name in metadata_path, "typo for input_path key"
                self.RAG_index[task_name]=faiss.read_index(vector_db_path[task_name])
                with open(metadata_path[task_name], 'r') as f:
                    self.metadata[task_name]= json.load(f)

    def preprocess_data(self, batch):
        def conver_to_template(conversation):
            """    
            from: <image>\nThere is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.
            to: prompt= 'USER: <image>\nPlease give me a one sentence caption about the image. ASSISTANT:'
            """    
            template=f'USER: {conversation} ASSISTANT:'
            return template
        image=[]
        text=[]
        image_names=[]
        
        for data in batch:
            process_text=conver_to_template(data['conversations'][0]['value'])
            # print("================")
            # print(process_text)
            image.append(data['image'])
            text.append(process_text)
            image_names.append(data['id'])
        inputs = self.processor(images=image, text=text,  padding=True, return_tensors='pt')        
        # inputs['ids']=image_name
        # print(inputs['pixel_values'].shape)
        return inputs, image_names


    def preprocess_data_prompt_tuning(self, batch):
        def conver_to_template(task, conversation):
            """    
            from: <image>\nThere is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.
            to: prompt= 'USER: <image>\nPlease give me a one sentence caption about the image. ASSISTANT:'
            """    
            prompt=""
            prompt_charactor="You are a traffic analysis assistant. "
            if 'general' in task:
                prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
                prompt_command="Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior."
                prompt=prompt_charactor+prompt_constrain+prompt_command
                
            elif 'suggestion' in task:
                """
                There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.
                """
                prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
                prompt_command= "Please provide driving suggestions for the ego car based on the current scene."
                prompt=prompt_charactor+prompt_constrain+prompt_command
                
            elif 'regional' in task:
                """
                Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.
                """
                prompt_constrain="There is an image of traffic captured from the perspective of the ego car. "
                prompt_command="Please describe the object inside the red rectangle in the image and explain why it affect ego car driving."
                prompt=prompt_charactor+prompt_constrain+prompt_command
                # print(prompt)
            
            template=f'USER: <image>\n{prompt} ASSISTANT:'
            return template
            # ---------------------------------------------------- zer's prompt ----------------------------------------------------
            # prompt=""
            # prompt_charactor="You are a traffic analysis assistant. "
            # if 'general' in task:
            #     prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
            #     prompt_command="Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior."
            #     prompt=prompt_charactor+prompt_constrain+prompt_command
                
            # elif 'suggestion' in task:
            #     """
            #     There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.
            #     """
            #     prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
            #     prompt_command= "Please provide driving suggestions for the ego car based on the current scene."
            #     prompt=prompt_charactor+prompt_constrain+prompt_command
                
            # elif 'regional' in task:
            #     """
            #     Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.
            #     """
            #     prompt_constrain="There is an image of traffic captured from the perspective of the ego car. "
            #     prompt_command="Please describe the object inside the red rectangle in the image and explain why it affect ego car driving."
            #     prompt=prompt_charactor+prompt_constrain+prompt_command
            #     # print(prompt)
            
            # template=f'USER: <image>\n{prompt} ASSISTANT:'
            # return template
        # ==========================================================================================================================================================================================
        image=[]
        text=[]
        image_names=[]
        
        for data in batch:
            process_text=conver_to_template(data['id'], data['conversations'][0]['value'], )
            # print(process_text)
            image.append(data['image'])
            text.append(process_text)
            image_names.append(data['id'])
        inputs = self.processor(images=image, text=text,  padding=True, return_tensors='pt')        
        return inputs, image_names
    
    def llava_preprocess_conversation(self, retreive_conversation):
        
        preprocess_text = "Here is a description regarding a driving scene. Please read the entire desciption and condense it into a single paragraph of non-bullet-point text."
        input_prompt = preprocess_text + retreive_conversation
        model_input = self.tokenizer(text=[input_prompt],  padding=True, return_tensors='pt') 
        condensed_output = self.llava_model.generate(**model_input, max_new_tokens=200, num_beams=1, do_sample=False)
        condensed_conversation = self.processor.decode(condensed_output[0][2:], skip_special_tokens=True)
        # print("condensed_conversation: ", condensed_conversation)

        return condensed_conversation
        
    
    def preprocess_data_RAG_prompt_tuning(self, batch):
        def conver_to_template(task, retrieve_conversation):
            """    
            from: <image>\nThere is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.
            to: prompt= 'USER: <image>\nPlease give me a one sentence caption about the image. ASSISTANT:'
            """

            cleaned_conversation = retrieve_conversation.replace("\n", "").replace("\r", "")
            prompt_charactor = "You are a traffic analysis assistant. "
            prompt_reference = "Here is a similar example for the following task, \"" + cleaned_conversation + "\""
            No_Plagiarism = "Use the above reference only as an example. It may not exactly match the current image. If there is any discrepancy, follow the actual image you see."
            
            if 'general' in task:
                prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
                prompt_command=" Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior."
                prompt=prompt_charactor+prompt_constrain+prompt_reference+No_Plagiarism+prompt_command
                
            elif 'suggestion' in task:
                """
                There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.
                """
                prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
                prompt_command= " Please provide driving suggestions for the ego car based on the current scene."
                prompt=prompt_charactor+prompt_constrain+prompt_reference+No_Plagiarism+prompt_command
                
            elif 'regional' in task:
                """
                Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.
                """
                prompt_command=" Please describe the object inside the red rectangle in the image and explain why it affect ego car driving."
                prompt=prompt_charactor+prompt_reference+No_Plagiarism+prompt_command
                
            template=f'USER: <image>\n{prompt} ASSISTANT:'
            return template
            
            # ---------------------------------------------------- chu's prompt ---------------------------------------------------- 
            # if 'general' in task:
            #     prompt_constrain = (
            #         "There is an image of traffic captured from the perspective of the ego car. "
            #         "Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), "
            #         "vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), "
            #         "traffic lights (red, green, yellow), traffic cones, barriers, "
            #         "miscellaneous(debris, dustbin, animals, etc.). "
            #         "You must not discuss any objects beyond the seven categories above. "
            #     )
            #     prompt_command = "Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.\n"
            # elif 'suggestion' in task:
            #     """s
            #     There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.
            #     """
            #     prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
            #     prompt_command= "Please provide driving suggestions for the ego car based on the current scene."
                
                
            # elif 'regional' in task:
            #     """
            #     Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.
            #     """
            #     prompt_constrain=""
            #     prompt_command="Please describe the object inside the red rectangle in the image and explain why it affect ego car driving."

            # prompt_context = (
            #     "Please refer to the above reference for style or details, but do not copy it exactly, and always base your description on the current image only."
            # )

            # prompt = (
            #     f"{prompt_charactor}{prompt_reference}{prompt_constrain}"
            #     f"{prompt_command}{prompt_context}"
            # )
            # template = f"USER: <image>\n{prompt} ASSISTANT:"
            # return template

            # -------------------------------------------- 阿哲 --------------------------------------------------------------------------
            # # 移除所有的換行符號
            # cleaned_conversation = retrieve_conversation.replace("\n", "").replace("\r", "")

            # prompt=""
            # prompt_charactor="You are a traffic analysis assistant. "
            # prompt_context="Here is a similar example for the following task, \"" + cleaned_conversation + "\""
            # # prompt_context = ""

            # # TODO: process three type of template 
            # if 'general' in task:
            #     prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
            #     prompt_command=" Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior."
            #     prompt=prompt_charactor+prompt_constrain+prompt_context+prompt_command
                
            # elif 'suggestion' in task:
            #     """
            #     There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.
            #     """
            #     prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
            #     prompt_command= " Please provide driving suggestions for the ego car based on the current scene."
            #     prompt=prompt_charactor+prompt_constrain+prompt_context+prompt_command
                
            # elif 'regional' in task:
            #     """
            #     Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.
            #     """
            #     prompt_command=" Please describe the object inside the red rectangle in the image and explain why it affect ego car driving."
            #     prompt=prompt_charactor+prompt_context+prompt_command
                
            # template=f'USER: <image>\n{prompt} ASSISTANT:'
            # return template
        # ==========================================================================================================================================================================================
        
        def conver_to_template_no_RAG(task):
            """    
            from: <image>\nThere is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior.
            to: prompt= 'USER: <image>\nPlease give me a one sentence caption about the image. ASSISTANT:'
            """    
            prompt=""
            prompt_charactor="You are a traffic analysis assistant. "
            if 'general' in task:
                prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
                prompt_command="Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior."
                prompt=prompt_charactor+prompt_constrain+prompt_command
                
            elif 'suggestion' in task:
                """
                There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene.
                """
                prompt_constrain="There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. "
                prompt_command= "Please provide driving suggestions for the ego car based on the current scene."
                prompt=prompt_charactor+prompt_constrain+prompt_command
                
            elif 'regional' in task:
                """
                Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.
                """
                prompt_constrain="There is an image of traffic captured from the perspective of the ego car. "
                prompt_command="Please describe the object inside the red rectangle in the image and explain why it affect ego car driving."
                prompt=prompt_charactor+prompt_constrain+prompt_command
                # print(prompt)
            
            template=f'USER: <image>\n{prompt} ASSISTANT:'
            return template
        # ==========================================================================================================================================================================================
        image=[]
        text=[]
        image_names=[]
        for data in batch:
            task=data['id'].split('_')[1]
            
            # TODO: RAG retrieve data
            # step: preprocess test images into concatenated embeddings using Semantic SAM and Depth anything
            concated_embedding= embed_fn(torchvision.transforms.functional.to_pil_image(data['image'], mode='RGB'))  # TODO: embed_fn, embed test image into a concated embedding 
            
            # step: To get the most similiar image in the database
            distance, indice= retrieve_similar_images(concated_embedding, self.RAG_index[task], top_k=3)
            # print("indice: ", indice)
            # print("distance: ", distance)
            
            # step: To get the meta data of the image 
            top_1=indice[0]
            top_1_dist=distance[0]
            # print("top_1 index: ", top_1)

            # condenced_conversation = self.llava_preprocess_conversation(retrieved_data)
            
            # step: Prompt tuning 
            # process_text=conver_to_template(data['id'], condenced_conversation)
            if top_1_dist < 10:
                retrieved_data=self.metadata[task][str(top_1)]['conversations']
                process_text=conver_to_template(data['id'], retrieved_data)
            else:
                process_text=conver_to_template_no_RAG(data['id'])

            # print("process_text: ", process_text)
            # print("retrieved_data: ", retrieved_data)
            
            # step: append data into a list for processor to tokenize and transform
            image.append(data['image'])
            text.append(process_text)
            image_names.append(data['id']) # for output .json format
        
        # step: Using processor to tokenize and transform
        inputs = self.processor(images=image, text=text,  padding=True, return_tensors='pt')        
        return inputs, image_names

# if __name__=='__main__':
#     from datasets import load_dataset
#     dataset_test= load_dataset("ntudlcv/dlcv_2024_final1",split='test')
    