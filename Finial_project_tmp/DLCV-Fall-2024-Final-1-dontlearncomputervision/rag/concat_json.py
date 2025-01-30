import json
import numpy as np

# 讀取兩個 JSON 檔案
with open('original_images_val_embeddings_CLIP.json', 'r') as file1, open('depth_map_val_embeddings_CLIP.json', 'r') as file2:
    data1 = json.load(file1)  # 28810 x 512
    data2 = json.load(file2)  # 28810 x 512

data1 = np.array(data1).squeeze(1)
data2 = np.array(data2).squeeze(1)


# 確保兩個文件的形狀一致
if len(data1) != len(data2):
    raise ValueError("Two json file don't have the same length!")

# 合併
combined_data = np.hstack((data1, data2))  # 水平拼接，生成 28810 x 1024

# 將 numpy array 轉換為 Python 列表
combined_data = combined_data.tolist()

# 寫入新的 JSON 文件
with open('val_combined.json', 'w') as combined_file:
    json.dump(combined_data, combined_file, indent=4)

print("JSON files successfully merged into 5716 x 1024!")