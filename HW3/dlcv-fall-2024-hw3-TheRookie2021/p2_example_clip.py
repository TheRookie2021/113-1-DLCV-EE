import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("hw3_data/p1_data/images/val/000000000000.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["some donut", "yummy", "sweet"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    print(image_features.dtype, image_features.shape)
    print(image_features)
    text_features = model.encode_text(text)
    
    print(text_features.dtype,text_features.shape)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]