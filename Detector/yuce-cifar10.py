import numpy as np
import json
import os
import sys
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils
from torchvision import models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time, datetime
import torchattacks
from torchvision.utils import save_image
from utils import imshow, image_folder_custom_label, l2_distance
import matplotlib.pyplot as plt
import imageio
from resnet18 import ResNet18
from PIL import Image
import cv2
#random.seed(2020)
def yuce(ae_root):
    flag="str"

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    class_idx = json.load(open('cifa10_class_index.json'))
    idx2label = [class_idx[str(k)][0] for k in range(len(class_idx))]
    #idx2label = [class_idx[1][1]]
    #print(k)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    # 加载数据
    if flag=="str":
        imagnet_data = image_folder_custom_label(root=ae_root, transform=transform, idx2label=idx2label,flag=flag)

    else:
        imagnet_data ,imgpath  = image_folder_custom_label(root='F:/org', transform=transform, idx2label=idx2label,flag=flag)

    data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=5, shuffle=True)
    images, labels = iter(data_loader).next()


    # 加载Inception V3
    model = models.resnet50(num_classes=10)
    model_weight_path = "cifar10-resnet50.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0
        
    for images, labels in data_loader:
        total = total + 5
        labels = labels.to(device)
        images = images.to(device)
        outputs = model(images)
        #outputs = model(images)
        _, pre = torch.max(outputs.data, 1)
        
        correct += (pre == labels).sum()
        #print(pre,"-------------",labels)
    print(ae_root)
    print("total:",total)
    print("correct:",correct)
    print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
      

yuce('F:/ae/ex-cifar10/fgsm-0.03')

