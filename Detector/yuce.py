import numpy as np
import json
import os
import sys
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.resnet import resnet152
from torchvision.models.vgg import vgg19

import torchvision.utils
from torchvision import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time, datetime
import torchattacks
from torchvision.utils import save_image
from model import resnet50
from utils import imshow, image_folder_custom_label, l2_distance
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import cv2
random.seed(2020)
def yuce(ae_root,model_name):
    flag="str"

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    class_idx = json.load(open('imagenet_class_index.json'))
    idx2label = [class_idx[str(k)][0] for k in range(len(class_idx))]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    # load data
    if flag=="str":
        imagnet_data = image_folder_custom_label(root=ae_root, transform=transform, idx2label=idx2label,flag=flag)

    else:
        imagnet_data ,imgpath  = image_folder_custom_label(root='F:/org', transform=transform, idx2label=idx2label,flag=flag)

    data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=8, shuffle=True)
    images, labels = iter(data_loader).next()


    # load Inception V3
    model = model_name(pretrained=True)
    model = model.eval().to(device)
    correct = 0
    rror = 0
        
    for images, labels in data_loader:
        labels = labels.to(device)
        images = images.to(device)
        outputs = model(images)
        _, pre = torch.max(outputs.data, 1)
        correct += (pre == labels).sum()
        rror +=(pre != labels).sum();
    print("total:",(rror+correct))
    print("correct:",correct)
    print(model_name,"-------->",ae_root)
    print('Robust accuracy: %.2f %%' % (100*float(correct) / (rror+correct)))



model_name = models.resnet50
root = "F:/ae/ex-imagnet/resnet50/ae_uap/"

yuce(root,model_name=model_name) 
