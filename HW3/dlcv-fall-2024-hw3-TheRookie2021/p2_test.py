# # from urllib.request import urlopen
# # from PIL import Image
# # import timm
# # import torch
# # from p2_utils import VitCaptionModel
# # # img = Image.open(urlopen(
# # #     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
# # # ))

# # model= VitCaptionModel(decoder_ckpt="hw3_data/p2_data/decoder_model.bin", lora_rank=0).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    
# # # model = model.eval()

# # # get model specific transforms (normalization, resize)
# # # data_config = timm.data.resolve_model_data_config(model)
# # # transforms = timm.data.create_transform(**data_config, is_training=False)

# # # output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1
# # optimizer = torch.optim.Adam( model.img_projection_layer.parameters(), lr=0.1)
# # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# # for o in range(20):
# #     scheduler.step()
# #     print(o,float(scheduler.get_last_lr()[0]))
# #     # print shape of each feature map in output
# #     # e.g.:
# #     #  torch.Size([1, 640, 24, 24])
# #     #  torch.Size([1, 640, 24, 24])
# #     #  torch.Size([1, 640, 24, 24])

# import json
# from p2_utils import *
# annotaion_path="hw3_data/p2_data/train.json"
# # note: parse json file into a sorted list 
# with open(annotaion_path) as f:
#     data = json.load(f)
# for k in data:
#     print(k)
# # annotation=data['annotations']

# # annotation=[(pair['image_id'], pair['caption']) for pair  in annotation ]
# # print(len(annotation))
# # print(annotation[19000:19005])
# # annotation=sorted(annotation)
# # print(annotation[19000:19005])


# images=data['images']
# print(len(images))
# image_id_mapping={}
# for pair  in images:
#     image_id_mapping[pair['id']]= pair['file_name']
# print(image_id_mapping[12])
# # captions=parseJSON(annotaion_path) 
# # print(len(captions))
# # print(captions[0])

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open("hw3_data/p2_data/images/train/000000000000.jpg")
im = im.resize((224, 224), Image.ANTIALIAS)
mask = np.zeros((224,224))
mask[30:-30, 30:-30] = 1 # white square in black background
# im = mask + np.random.randn(10,10) * 0.01 # random image

masked = np.ma.masked_where(mask == 0, mask)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(im, 'gray', interpolation='none')
plt.subplot(1,2,2)
plt.imshow(im, 'gray', interpolation='none')
plt.imshow(masked, 'jet', interpolation='none', alpha=0.2)
plt.savefig("test_attmap/test_overlap.png")
