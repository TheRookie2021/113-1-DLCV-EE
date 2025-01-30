import gdown

lora_adaptater_model_checkpoint_link = "https://drive.google.com/drive/folders/1EIDOsibRG5WPQngyJnon6eYtb6M5S6tT?usp=sharing"
lora_adaptater_model_checkpoint_path = "./finetune_llava-1.5-7b-hf_lora_5_weight_decay/checkpoint-86430"

metadata_link = "https://drive.google.com/drive/folders/1oC-M3txMSZgEW1zharWI969c3_c90y9r?usp=sharing"
metadata_path = "./inference/metadata"

model_checkpoint_link = "https://drive.google.com/drive/folders/18sR_e7n5O6Wf8EvYo7V7GH4WRCeTUw2B?usp=sharing"
model_checkpoint_path = "./inference/model_checkpoints"

RAG_dataset_link = "https://drive.google.com/drive/folders/1XQP0buyCUqQHn0D0DZKQWKiRH7Q7f21v?usp=sharing"
RAG_dataset_path = "./inference/RAG_dataset"

gdown.download_folder(lora_adaptater_model_checkpoint_link, output=lora_adaptater_model_checkpoint_path)
gdown.download_folder(metadata_link, output=metadata_path)
gdown.download_folder(model_checkpoint_link, output=model_checkpoint_path)
gdown.download_folder(RAG_dataset_link, output=RAG_dataset_path)

