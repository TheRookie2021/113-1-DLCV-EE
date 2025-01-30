from transformers import ViTModel, ViTFeatureExtractor
import torch
from PIL import Image
import os

# 加載模型和特徵提取器
model = ViTModel.from_pretrained("google/vit-base-patch16-224")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# 圖片嵌入提取函數
def extract_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)  # 平均池化
    return embedding.squeeze().numpy()

# 批量提取圖片嵌入
embeddings = []
image_paths = []
for image_file in os.listdir("path/to/images"):
    embedding = extract_embedding(os.path.join("path/to/images", image_file))
    embeddings.append(embedding)
    image_paths.append(image_file)

import faiss
import numpy as np

# 將嵌入轉換為 NumPy 數組
embeddings_array = np.stack(embeddings)

# 建立 FAISS 索引
d = embeddings_array.shape[1]  # 嵌入的維度
index = faiss.IndexFlatL2(d)
index.add(embeddings_array)

# 保存索引
faiss.write_index(index, "image_embeddings.index")

# 保存圖片對應的編號
with open("image_paths.txt", "w") as f:
    f.writelines([f"{path}\n" for path in image_paths])

query_embedding = extract_embedding("path/to/query_image.jpg")


# 加載索引
index = faiss.read_index("image_embeddings.index")

# 執行最近鄰檢索
k = 1  # 檢索前 5 名
distances, indices = index.search(query_embedding[np.newaxis, :], k)

# 獲取對應的圖片編號
with open("image_paths.txt", "r") as f:
    image_paths = f.readlines()
result_images = [image_paths[i].strip() for i in indices[0]]
print("最相似的圖片：", result_images)
