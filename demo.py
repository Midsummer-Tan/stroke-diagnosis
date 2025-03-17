import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image

import utils
from model import model_mymodel

model = model_mymodel.MyModel()
model_path = './save/model/epoch-latest.pth'
# './save/model/best-dice-model.pth' or './save/model/epoch-latest.pth'
if model_path.split('/')[-1] == 'epoch-latest.pth':
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
else:
    model.load_state_dict(torch.load(model_path))
model.eval()
img_path = "../stroke-diagnosis/load/ASID_v2/test/image"
save_path = "./save/img"
name_list = os.listdir(img_path)


for i in range(len(name_list)):
    image_1 = transforms.ToTensor()(Image.open(os.path.join(img_path, name_list[i])))
    if i==0 or name_list[i-1][0:3]!=name_list[i][0:3] or \
        int(name_list[i-1][3:6])!=int(name_list[i][3:6])-1:
        image_0 = image_1
    else:
        image_0 = transforms.ToTensor()(Image.open(os.path.join(img_path, name_list[i-1])))
    if i == len(name_list)-1 or name_list[i+1][0:3]!=name_list[i][0:3] or \
        int(name_list[i+1][3:6])!=int(name_list[i][3:6])+1:
        image_2 = image_1
    else:
        image_2 = transforms.ToTensor()(Image.open(os.path.join(img_path, name_list[i+1])))
    image = torch.squeeze(torch.stack([image_0,image_1,image_2])).unsqueeze(0)
    pred = torch.sigmoid(model(image)[:,0].squeeze())
    w, h = pred.size()
    new_mask_array = np.zeros((w, h))
    for p in range(w):
        for j in range(h):
            if pred[p,j].item() > 0.5:
                new_mask_array[p,j] = 255
    new_mask = Image.fromarray(new_mask_array)
    new_mask.convert('1').save(os.path.join(save_path, name_list[i]))
