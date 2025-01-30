# -------------------------------------Training -------------------------------------------------
import clip
import json
import faiss
import torch
from PIL import Image
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

all_embeddings = []

for i in range(28810):
    print(f"CLIP: Processing the {i}th segment train image.")
    image = preprocess(Image.open(f"../yolo/segment_output_images/train/train_segment_{i}.jpg")).unsqueeze(0).to(device)
    # print(type(image))  # torch.Tensor
    # print(image.shape)  # torch.Size([1, 3, 224, 224])

    with torch.no_grad():
        image_features = model.encode_image(image)
        # print(type(image_features))  # <class 'torch.Tensor'>
        # print(image_features.shape)  # torch.Size([1, 512])

    image_features_2list = image_features.tolist()  # list
    all_embeddings.append(image_features_2list)
    print(image_features_2list)

    # image_features_2np = image_features.cpu().numpy()  # numpy.ndarray

    # if i == 0:
    #     embedding_means = image_features_2np.reshape(1, -1)
    # else:
    #     embedding_mean = image_features_2np.reshape(1, -1)
    #     embedding_means = np.append(embedding_means, embedding_mean, axis=0)  # numpy.ndarray


# ------------------------------------------------------------------------ Save the embeddings into json ------------------------------------------------------------------------
# Define the output JSON file path
output_json_path = "./segment_train_embeddings_CLIP.json"

# Save embeddings to a JSON file
with open(output_json_path, "w") as json_file:
    json.dump(all_embeddings, json_file)

print(f"Embeddings saved to {output_json_path}")




# # ------------------------------------- Validation -------------------------------------------------
# import clip
# import json
# import faiss
# import torch
# from PIL import Image
# import numpy as np


# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# all_embeddings = []

# for i in range(8716):
#     print(f"CLIP: Processing the {i}th segment validation image.")
#     image = preprocess(Image.open(f"../yolo/segment_output_images/val/val_segment_{i}.jpg")).unsqueeze(0).to(device)
#     # print(type(image))  # torch.Tensor
#     # print(image.shape)  # torch.Size([1, 3, 224, 224])

#     with torch.no_grad():
#         image_features = model.encode_image(image)
#         # print(type(image_features))  # <class 'torch.Tensor'>
#         # print(image_features.shape)  # torch.Size([1, 512])

#     image_features_2list = image_features.tolist()  # list
#     all_embeddings.append(image_features_2list)

#     # image_features_2np = image_features.cpu().numpy()  # numpy.ndarray

#     # if i == 0:
#     #     embedding_means = image_features_2np.reshape(1, -1)
#     # else:
#     #     embedding_mean = image_features_2np.reshape(1, -1)
#     #     embedding_means = np.append(embedding_means, embedding_mean, axis=0)  # numpy.ndarray


# # ------------------------------------------------------------------------ Save the embeddings into json ------------------------------------------------------------------------
# # Define the output JSON file path
# output_json_path = "./segment_val_embeddings_CLIP.json"

# # Save embeddings to a JSON file
# with open(output_json_path, "w") as json_file:
#     json.dump(all_embeddings, json_file)

# print(f"Embeddings saved to {output_json_path}")




# # -------------------------------------Training -------------------------------------------------
# import clip
# import json
# import faiss
# import torch
# from PIL import Image
# import numpy as np


# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# all_embeddings = []

# for i in range(28810):
#     print(f"CLIP: Processing the {i}th bbox train image.")
#     image = preprocess(Image.open(f"../yolo/bbox_output_images/train/train_bbox_{i}.jpg")).unsqueeze(0).to(device)
#     # print(type(image))  # torch.Tensor
#     # print(image.shape)  # torch.Size([1, 3, 224, 224])

#     with torch.no_grad():
#         image_features = model.encode_image(image)
#         # print(type(image_features))  # <class 'torch.Tensor'>
#         # print(image_features.shape)  # torch.Size([1, 512])

#     image_features_2list = image_features.tolist()  # list
#     all_embeddings.append(image_features_2list)

#     # image_features_2np = image_features.cpu().numpy()  # numpy.ndarray

#     # if i == 0:
#     #     embedding_means = image_features_2np.reshape(1, -1)
#     # else:
#     #     embedding_mean = image_features_2np.reshape(1, -1)
#     #     embedding_means = np.append(embedding_means, embedding_mean, axis=0)  # numpy.ndarray


# # ------------------------------------------------------------------------ Save the embeddings into json ------------------------------------------------------------------------
# # Define the output JSON file path
# output_json_path = "./bbox_images_train_embeddings_CLIP.json"

# # Save embeddings to a JSON file
# with open(output_json_path, "w") as json_file:
#     json.dump(all_embeddings, json_file)

# print(f"Embeddings saved to {output_json_path}")





# ------------------------------------- Validation -------------------------------------------------
import clip
import json
import faiss
import torch
from PIL import Image
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

all_embeddings = []

for i in range(8716):
    print(f"CLIP: Processing the {i}th bbox validation image.")
    image = preprocess(Image.open(f"../yolo/bbox_output_images/val/val_bbox_{i}.jpg")).unsqueeze(0).to(device)
    # print(type(image))  # torch.Tensor
    # print(image.shape)  # torch.Size([1, 3, 224, 224])

    with torch.no_grad():
        image_features = model.encode_image(image)
        # print(type(image_features))  # <class 'torch.Tensor'>
        # print(image_features.shape)  # torch.Size([1, 512])

    image_features_2list = image_features.tolist()  # list
    all_embeddings.append(image_features_2list)

    # image_features_2np = image_features.cpu().numpy()  # numpy.ndarray

    # if i == 0:
    #     embedding_means = image_features_2np.reshape(1, -1)
    # else:
    #     embedding_mean = image_features_2np.reshape(1, -1)
    #     embedding_means = np.append(embedding_means, embedding_mean, axis=0)  # numpy.ndarray


# ------------------------------------------------------------------------ Save the embeddings into json ------------------------------------------------------------------------
# Define the output JSON file path
output_json_path = "./bbox_images_val_embeddings_CLIP.json"

# Save embeddings to a JSON file
with open(output_json_path, "w") as json_file:
    json.dump(all_embeddings, json_file)

print(f"Embeddings saved to {output_json_path}")