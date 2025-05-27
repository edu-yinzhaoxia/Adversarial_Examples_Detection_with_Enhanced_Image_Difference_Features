import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import torchvision.datasets as dsets


def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize=(5, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def image_folder_custom_label(root, transform, idx2label,flag):
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']

    old_data = dsets.ImageFolder(root=root, transform=transform)
    if flag == "str":
        old_classes = old_data.classes
        label2idx = {}

        for i, item in enumerate(idx2label):
            label2idx[item] = i

        new_data = dsets.ImageFolder(root=root, transform=transform,
                                     target_transform=lambda x: idx2label.index(old_data.classes[x]))
        # new_data = dsets.ImageFolder(root=root, transform=transform)
        new_data.classes = idx2label
        new_data.class_to_idx = label2idx
        return new_data
    
    else:
        class_idx = json.load(open('imagenet_class_index.json'))
        
        old_classes = class_idx[str(int(old_data.classes[0]))][1]   
        old_data.classes = old_classes   
        label2idx = {}
        #print(old_data.tar)
        for i, item in enumerate(idx2label):
            label2idx[item] = i

        new_data = dsets.ImageFolder(root=root, transform=transform,
                                     target_transform=None)
        #new_data = dsets.ImageFolder(root=root, transform=transform)
        new_data.classes = idx2label
        new_data.class_to_idx = label2idx
        # print(old_data.imgs)
        return new_data, old_data.imgs


def l2_distance(outputs, images, adv_images, labels, device="cuda"):
    outputs = outputs
    _, pre = torch.max(outputs.data, 1)
    corrects = (labels.to(device) == pre)
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2
'''
Functions for:
- Loading models, datasets
- Evaluating on datasets with or without UAP
'''
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import json
import torch
from torch._C import device
import torch.nn as nn
import torchvision
from PIL import Image
import imageio
from torch.utils import model_zoo
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

# Use the built-in transforms functions from torchvision in the loader
loader = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))]) 
 
unloader = transforms.ToPILImage()


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

def image_loader(image_name,device):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

