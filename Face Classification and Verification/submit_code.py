#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import os
import numpy as np
from PIL import Image

import torch
import torchvision   
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import logging

from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
import pandas as pd


# ## Residual Block
# 
# Resnet: https://arxiv.org/pdf/1512.03385.pdf
# 
# Here is a basic usage of shortcut in Resnet

# In[54]:


class BasicBlock(nn.Module):

    def __init__(self, channel_size, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.shortcut = nn.Conv2d(channel_size, channel_size, kernel_size=1, stride=stride, bias=False)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(out))
        x = self.shortcut(x)
#         print("out: ", str(out.size()))
#         print("x: ", str(x.size()))
        out += x
        out = F.relu(out)
        #print(out.size())
        #print('***********')
        return out


# ## CNN Model with Residual Block 

# In[55]:


class Network(nn.Module):
    def __init__(self, num_feats, hidden_sizes, num_classes, feat_dim=10):
        super(Network, self).__init__()
        self.hidden_sizes = [num_feats] + hidden_sizes + [num_classes]
        self.layers = []
        self.layers.append(nn.Conv2d(in_channels=3, out_channels= 32, kernel_size=3, stride=1, bias=False))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(BasicBlock(channel_size = 32))
                           
        self.layers.append(nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=3, stride=2, bias=False))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(BasicBlock(channel_size = 64))
        
        self.layers.append(nn.Conv2d(in_channels=64, out_channels= 128, kernel_size=3, stride=2, bias=False))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(BasicBlock(channel_size = 128))
        
        self.layers.append(nn.Conv2d(in_channels=128, out_channels= 256, kernel_size=3, stride=1, bias=False))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(BasicBlock(channel_size = 256))
        
        self.layers.append(nn.Conv2d(in_channels=256, out_channels= 512, kernel_size=3, stride=1, bias=False))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(BasicBlock(channel_size = 512))
                           
        self.layers.append(nn.Conv2d(in_channels=512, out_channels= 1024, kernel_size=3, stride=1, bias=False))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(BasicBlock(channel_size = 1024))
                           
        self.layers.append(nn.Conv2d(in_channels=1024, out_channels= 512, kernel_size=3, stride=1, bias=False))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(BasicBlock(channel_size = 512))
        
        self.layers = nn.Sequential(*self.layers)
        
        self.emb_layer = nn.Linear(self.hidden_sizes[-2], 512, bias=False)
        self.linear_label = nn.Linear(512, self.hidden_sizes[-1], bias=True)
        self.dp = nn.Dropout(0.3)
        
        # For creating the embedding to be passed into the Center Loss criterion
        self.linear_closs = nn.Linear(self.hidden_sizes[-2], feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)
    
    def forward(self, x, evalMode=False):
        output = x # (B, 3, W, H)
#         print(output.size())
        output = self.layers(output)  # (B, 512, k, k)\
        #print(output.size())
        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1) # b 512 1 
        output = output.reshape(output.shape[0], output.shape[1]) #b 512
        output_emb = self.emb_layer(output) # 512 512
        output = self.dp(output_emb)
        label_output = self.linear_label(output) # 512 4000
        label_output = label_output/torch.norm(self.linear_label.weight, dim=1)
        
        # Create the feature embedding for the Center Loss
        closs_output = self.linear_closs(output)
        closs_output = self.relu_closs(closs_output)

        return closs_output, label_output, output_emb

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)


# ### Training & Testing Model

# In[16]:



#start = 16
def train(model, data_loader, test_loader, task='Classification'):
    model.train()
    
    for epoch in range(numEpochs):
        start_time = time.time()
        
        avg_loss = 0.0
        for batch_num, (feats, labels) in tqdm(enumerate(data_loader)):
            feats, labels = feats.to(device), labels.to(device)
#             print("feats: ", str(feats.shape))
            optimizer.zero_grad()
            outputs = model(feats)[1] # _call function, implement forward
            

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()

            if batch_num % 1000 == 999:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/1000))
                avg_loss = 0.0    
            
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
        
        if task == 'Classification':
            val_loss, val_acc = test_classify(model, test_loader)
            train_loss, train_acc = test_classify(model, data_loader)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc, val_loss, val_acc))
        else:
            test_verify(model, test_loader)
        end_time = time.time()
        print('Time: ',end_time - start_time, 's')
        print('='*20)
    
        # may save the training model for future use
        if not os.path.exists("./model"):
            os.mkdir("./model")

        torch.save(model.state_dict(),'/home/ubuntu/11785/hw2/model/{}_{}_{}'.format('MODEL_NAME',epoch, 'flip'))
        logging.info('model saved to ./model/{}_{}_{}'.format('MODEL_NAME',epoch,'flip'))
        
        res = get_cosine_simil(model, val_veri_dataloader)
        print('AUC score', roc_auc_score(np.asarray(label_list), res))


def test_classify(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)[1]
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        loss = criterion(outputs, labels.long())
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total


def test_verify(model, test_loader):
    raise NotImplementedError


# #### Dataset, DataLoader and Constant Declarations

# In[4]:


data_transform_test = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                      torchvision.transforms.RandomVerticalFlip(p=0.5),
                                                      torchvision.transforms.ToTensor()])


# In[5]:


train_dataset = torchvision.datasets.ImageFolder(root='/home/ubuntu/11785/hw2/classification_data/train_data/', 
                                                 transform=data_transform_test )
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, 
                                               shuffle=True, num_workers=0)

dev_dataset = torchvision.datasets.ImageFolder(root='/home/ubuntu/11785/hw2/classification_data/val_data/', 
                                               transform=torchvision.transforms.ToTensor())
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=64, 
                                             shuffle=True, num_workers=0)


# In[6]:


numEpochs = 25
num_feats = 3

learningRate = 1e-3
weightDecay = 5e-5

num_classes = len(train_dataset.classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[7]:


network = Network(num_feats, hidden_sizes, num_classes)
network.to(device)
network.apply(init_weights)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(network.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)


# In[ ]:


# first train, after epoch 15, val accuracy stucks
network.train()
train(network, train_dataloader, dev_dataloader)


# In[8]:


# reload the first train, epoch 15 model, change lr to 1e-3
network.load_state_dict(torch.load("./model/MODEL_NAME_15"))
# second train based on first train epoch 15 model, trained 3 epoch
network.train()
train(network, train_dataloader, dev_dataloader)


# ### Custom DataSet with DataLoader

# In[8]:


class ImageDataset(Dataset):
    def __init__(self, file_list, file_list1):
        self.file_list = file_list
        self.file_list1 = file_list1
        #self.n_class = len(list(set(target_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        img1 = Image.open(self.file_list1[index])
        img1 = torchvision.transforms.ToTensor()(img1)
        #label = self.target_list[index]
        return img, img1


# In[9]:


verifi_txt = np.loadtxt('verification_pairs_val.txt', delimiter = ' ', dtype=bytes).astype(str)
image_list = verifi_txt[:,0]
image_list1 = verifi_txt[:,1]
label_list = verifi_txt[:,2]
val_veri_dataset = ImageDataset(image_list, image_list1)
val_veri_dataloader = DataLoader(val_veri_dataset, batch_size=4, shuffle=False, num_workers=0)


# In[10]:


verifi_txt_test = np.loadtxt('verification_pairs_test.txt', delimiter = ' ', dtype=bytes).astype(str)
image_list = verifi_txt_test[:,0]
image_list1 = verifi_txt_test[:,1]

test_veri_dataset = ImageDataset(image_list, image_list1)
test_veri_dataloader = DataLoader(test_veri_dataset, batch_size=4, shuffle=False, num_workers=0)


# In[11]:


def get_cosine_simil(model, dataloader):
    model.eval()
    result = []
    for batch_num, (img, img1) in tqdm(enumerate(dataloader)):
        img, img1 = img.to(device), img1.to(device)
        img_emb_feature = model(img)[2].detach() # batch_size * 512(# of neurons)
        img1_emb_feature = model(img1)[2].detach()  # batch_size * 512
        cos = nn.CosineSimilarity(dim=1)
        output = cos(img_emb_feature, img1_emb_feature) # batch_size 
        del img
        del img1
        del img_emb_feature
        del img1_emb_feature
        result.append(output.cpu())
        #result.append(output)
    
    results = torch.cat(result)
    results = results.numpy() # number of verification pairs
    
    return results


# In[14]:


# data augmentaion based on random horizontal flip and vertical flip
# third train, based on second train best model, trained 7 epoch
network.load_state_dict(torch.load("./model/MODEL_NAME_17"))
network.train()
train(network, train_dataloader, dev_dataloader)


# In[19]:


network.load_state_dict(torch.load("./model/MODEL_NAME_5_flip"))
test_res = get_cosine_simil(network, test_veri_dataloader)
import pandas as pd
sub = pd.read_csv('verification_pairs_test.txt', header = None)
submit = sub.set_axis(['Id'], axis=1, inplace=False)
submit['Category'] = test_res
submit.to_csv("submit.csv", index=False)


# ## Center Loss

# In[ ]:


class CenterLoss(nn.Module):
    """
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, device=torch.device('cpu')):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) +                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


# In[ ]:


def train_closs(model, data_loader, test_loader, task='Classification'):
    model.train()

    for epoch in range(numEpochs):
        avg_loss = 0.0
        for batch_num, (feats, labels) in tqdm(enumerate(data_loader)):
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer_label.zero_grad()
            optimizer_closs.zero_grad()
            
            feature, outputs, _ = model(feats)

            l_loss = criterion_label(outputs, labels.long())
            c_loss = criterion_closs(feature, labels.long())
            loss = l_loss + closs_weight * c_loss
            
            loss.backward()
            
            optimizer_label.step()
            # by doing so, weight_cent would not impact on the learning of centers
            for param in criterion_closs.parameters():
                param.grad.data *= (1. / closs_weight)
            optimizer_closs.step()
            
            avg_loss += loss.item()

            if batch_num % 1000 == 999:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/1000))
                avg_loss = 0.0    
            
            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
        
        if not os.path.exists("./model"):
            os.mkdir("./model")

        torch.save(model.state_dict(),'/home/ubuntu/11785/hw2/model/{}_{}_{}'.format('MODEL_NAME',epoch, 'center'))
        logging.info('model saved to ./model/{}_{}_{}'.format('MODEL_NAME',epoch,'center'))
        
        res = get_cosine_simil(model, val_veri_dataloader)
        print('AUC score', roc_auc_score(np.asarray(label_list), res))
        
        if task == 'Classification':
            val_loss, val_acc = test_classify_closs(model, test_loader)
            train_loss, train_acc = test_classify_closs(model, data_loader)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
                  format(train_loss, train_acc, val_loss, val_acc))
        else:
            test_verify(model, test_loader)


def test_classify_closs(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        feature, outputs, _ = model(feats)
        
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        l_loss = criterion_label(outputs, labels.long())
        c_loss = criterion_closs(feature, labels.long())
        loss = l_loss + closs_weight * c_loss
        
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total


# In[ ]:


closs_weight = 1
lr_cent = 0.01
feat_dim = 10

network = Network(num_feats, hidden_sizes, num_classes, feat_dim)
network.apply(init_weights)

criterion_label = nn.CrossEntropyLoss()
criterion_closs = CenterLoss(num_classes, feat_dim, device)
optimizer_label = torch.optim.SGD(network.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9)
optimizer_closs = torch.optim.SGD(criterion_closs.parameters(), lr=lr_cent)


# In[ ]:


network.load_state_dict(torch.load("./model/MODEL_NAME_5_flip"))
network.train()
network.to(device)
train_closs(network, train_dataloader, dev_dataloader)

