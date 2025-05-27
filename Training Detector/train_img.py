import os
import json
import random
from sqlalchemy import false
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import resnet50
from tqdm import tqdm
import platform
import argparse
import os 
import numpy as np
random.seed(2020)

def main():
 
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='PyTorch  Training')
    parser.add_argument('--outf', default='F:/coding/ClassificationModels/pytorch_classification/Test5_resnet/', help='folder to output images and model checkpoints') 
    args = parser.parse_args()
    net = resnet50(pretrained=false)

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "F:/ae/btws/imagenet/resnet50/"))  # get data root path
    image_path = os.path.join(data_root, "uap_btws")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=2)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 0])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=4)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=4)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    


    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 2)
    net.to(device)
    

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 300
    best_acc = 0.0
    save_path = "uap_btws-imagenet-resnet50-erfen.pth"
    train_steps = len(train_loader)
    if not os.path.exists(args.outf):
		    os.makedirs(args.outf)
      
    print("Start Training, Resnet50!")
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(epochs):
                # train
                net.train()
                running_loss = 0.0
                train_bar = tqdm(train_loader)
                for i, data in enumerate(train_bar):
                    length = len(train_loader)
                    images, labels = data
                    optimizer.zero_grad()
                    logits = net(images.to(device))
                    loss = loss_function(logits, labels.to(device))
                    loss.backward()
                    optimizer.step()
        
                    # print statistics
                    running_loss += loss.item()
        
                    train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                             epochs,
                                                                             loss)
                    print('[epoch %d] train_loss: %.3f' %
                      (epoch + 1, running_loss / train_steps))
                    print('[epoch:%d, iter:%d] Loss: %.03f '
                          % (epoch + 1, (i + 1 + epoch * length), running_loss / (i + 1)))
                    f2.write('%03d  %05d |Loss: %.03f '
                          % (epoch + 1, (i + 1 + epoch * length), running_loss / (i + 1)))
                    f2.write('\n')
                    f2.flush()
        
                # validate
                net.eval()
                acc = 0.0  # accumulate accurate number / epoch
                with torch.no_grad():
                    val_bar = tqdm(validate_loader)
                    for val_data in val_bar:
                        val_images, val_labels = val_data
                        outputs = net(val_images.to(device))
                        # loss = loss_function(outputs, test_labels)
                        predict_y = torch.max(outputs, dim=1)[1]
                        acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        
                        val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                                   epochs)
        
                val_accurate = acc / val_num
                
                if val_accurate > best_acc:
                    best_acc = val_accurate
                    torch.save(net.state_dict(), save_path)
                    f3 = open("best_acc.txt", "w")
                    f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, best_acc))
                    f3.close()
                    print(best_acc)
                    
                 
            print('Finished Training')
            print(best_acc)

if __name__ == '__main__':
    main()
