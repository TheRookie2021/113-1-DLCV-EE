import json
import faiss
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from datasets import load_dataset
# ------------------------------------------------------------------------ InternViT ------------------------------------------------------------------------
model = AutoModel.from_pretrained(
    'OpenGVLab/InternViT-6B-224px',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).cuda().eval()

image_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-6B-224px')
dataset_test= load_dataset("ntudlcv/dlcv_2024_final1",split='train')
dataset_test= dataset_test.with_format("torch")

all_embeddings = []

for i, data in enumerate(dataset_test):
# for i in range(5):
    if i==5 : break
    # print("round:", i)
    # image = Image.open(f'./input/image{i}.jpg').convert('RGB')

    # print(type(image))

    image = data['image']
    print(image.shape) # torch.Size([3, 720, 1355])
    print(type(image)) # <class 'torch.Tensor'>

    
    pixel_values = image_processor(images=image, return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    outputs = model(pixel_values)
    # print(outputs)

    embedding = outputs.last_hidden_state.to(dtype=torch.float32)
    # print(embeddings.type())  # torch.cuda.FloatTensor

    # Convert embeddings tensor to a Python list for JSON serialization
    embedding_list = embedding.detach().cpu().numpy().tolist()  # list

    # 去除 batch 維度
    embedding_list = embedding_list[0]                          # list
    embedding_mean = np.mean(embedding_list, axis = 0)          # numpy.ndarray
    all_embeddings.append(embedding_mean.tolist())              # list

    if i == 0:
        embedding_means = embedding_mean.reshape(1, -1)
    else:
        embedding_mean = embedding_mean.reshape(1, -1)
        embedding_means = np.append(embedding_means, embedding_mean, axis=0)  # numpy.ndarray

# print("\n")
# print(type(embedding_means))  # numpy.ndarray
# print(embedding_means.shape)  # (5, 3200)

# print(type(all_embeddings))   # ist
# print("row num:", len(all_embeddings))
# print("col num:", len(all_embeddings[0]))
# print("\n")


# ------------------------------------------------------------------------ Save the embeddings into json ------------------------------------------------------------------------
# Define the output JSON file path
output_json_path = "./embeddings_InternViT.json"

# Save embeddings to a JSON file
with open(output_json_path, "w") as json_file:
    json.dump(all_embeddings, json_file)

print(f"Embeddings saved to {output_json_path}")


# ------------------------------------------------------------------------ FAISS ------------------------------------------------------------------------
# Set dimensions and number of data points
dimension = 3200
database_size = 5

# Initialize FAISS index
index = faiss.IndexFlatL2(dimension)        # L2 distance
# index = faiss.IndexFlatIP(dimension)      # inner product
# index = faiss.IndexBinaryFlat(dimension)  # Hamming distance
print("Is trained:", index.is_trained)

# Add vectors to the index
index.add(embedding_means)
print("Number of vectors in index:", index.ntotal)

# Query vector
query_vector = embedding_means[1].reshape(1, -1)

# Search for the 5 closest vectors
search_closest_num = 2
distances, indices = index.search(query_vector, search_closest_num)

# Print the distances to the nearest neighbors
print("Distances:", distances)
# Print the indices of the nearest neighbors
print("Indices:", indices)


