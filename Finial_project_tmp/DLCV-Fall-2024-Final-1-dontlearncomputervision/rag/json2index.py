# convert json file to faiss index file

import faiss
import numpy as np
import json
import os

input_json_path = "depth_map_train_embeddings_CLIP.json"

with open(input_json_path, "r") as f:
    db = json.load(f)
# print(np.array(db).squeeze(1).shape)
db = np.array(db).squeeze(1)


# Initialize FAISS index
dimension = 512
index = faiss.IndexFlatL2(dimension)        # L2 distance
# index = faiss.IndexFlatIP(dimension)      # inner product
# index = faiss.IndexBinaryFlat(dimension)  # Hamming distance
# print("Is trained:", index.is_trained)

# Add vectors to the index
index.add(db)
output_index_path = input_json_path[:-5] + ".index"
faiss.write_index(index, output_index_path)
print("Number of vectors in index:", index.ntotal)

saved_index=faiss.read_index(output_index_path)
print("Number of vectors in index:", saved_index.ntotal)


