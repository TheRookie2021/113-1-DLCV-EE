import clip
import json
import faiss
import torch
from PIL import Image
import numpy as np
from datasets import load_dataset


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

dataset = load_dataset("../data/dlcv_2024_final1", split='validation')
# dataset = dataset.with_format("torch")


all_embeddings = []

for i, data in enumerate(dataset):
    print(f"CLIP: Processing the {i}th original data (val).")

    image = data['image']
    image = preprocess(image).unsqueeze(0).to(device)
    # print(type(image))  # torch.Tensor
    # print(image.shape)  # torch.Size([1, 3, 224, 224])

    with torch.no_grad():
        image_features = model.encode_image(image)
        # print(type(image_features))  # <class 'torch.Tensor'>
        # print(image_features.shape)  # torch.Size([1, 512])

    image_features_2list = image_features.tolist()  # list
    all_embeddings.append(image_features_2list)

    image_features_2np = image_features.cpu().numpy()  # numpy.ndarray

    if i == 0:
        embedding_means = image_features_2np.reshape(1, -1)
    else:
        embedding_mean = image_features_2np.reshape(1, -1)
        embedding_means = np.append(embedding_means, embedding_mean, axis=0)  # numpy.ndarray


# ------------------------------------------------------------------------ Save the embeddings into json ------------------------------------------------------------------------
# Define the output JSON file path
output_json_path = "./original_images_val_embeddings_CLIP.json"

# Save embeddings to a JSON file
with open(output_json_path, "w") as json_file:
    json.dump(all_embeddings, json_file)

print(f"Embeddings saved to {output_json_path}")


# # ------------------------------------------------------------------------ FAISS ------------------------------------------------------------------------
# # Set dimensions and number of data points
# dimension = 512
# database_size = 5

# # Initialize FAISS index
# index = faiss.IndexFlatL2(dimension)        # L2 distance
# # index = faiss.IndexFlatIP(dimension)      # inner product
# # index = faiss.IndexBinaryFlat(dimension)  # Hamming distance
# print("Is trained:", index.is_trained)

# # Add vectors to the index
# index.add(embedding_means)
# print("Number of vectors in index:", index.ntotal)

# # Query vector
# query_vector = embedding_means[1].reshape(1, -1)

# # Search for the 5 closest vectors
# search_closest_num = 2
# distances, indices = index.search(query_vector, search_closest_num)

# # Print the distances to the nearest neighbors
# print("Distances:", distances)
# # Print the indices of the nearest neighbors
# print("Indices:", indices)
