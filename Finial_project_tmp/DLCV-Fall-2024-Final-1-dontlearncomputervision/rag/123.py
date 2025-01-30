import json
import numpy as np

# 讀取兩個 JSON 檔案
with open('depth_map_train_embeddings_CLIP.json', 'r') as file:
    data = json.load(file)  # 28810 x 512
    
print(len(data))
print((data[0][0]))