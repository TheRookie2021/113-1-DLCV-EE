from datasets import load_dataset

# # download dataset from hugging face
# dataset = load_dataset("ntudlcv/dlcv_2024_final1")
# dataset.save_to_disk("data/dlcv_2024_final1")

# load dataset from the given path
dataset = load_dataset("data/dlcv_2024_final1", split='train')

for i, data in enumerate(dataset):
    
    if i == 5: break

    print(data['id'])
    print(data['image'])
    print(data['conversations'])
