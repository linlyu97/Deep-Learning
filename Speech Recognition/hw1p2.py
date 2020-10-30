#!/usr/bin/env python
# coding: utf-8

# In[37]:


import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils import data
from torchvision import transforms
import os
import gc
import logging
import time


# In[38]:


batch_size = 1024
context = 35


# In[39]:


def data_load(X):
    X = np.load(X,allow_pickle=True)
    data1=[]
    for utter in X:
        data1.append(np.pad(utter, [(context,context),(0,0)], 'constant', constant_values=(0,0)))
    data1 = torch.from_numpy(np.concatenate(data1))
    return data1
def load(Y):
    Y = np.load(Y,allow_pickle=True)
    lab=[]
    for item in Y:
        lab.append(np.pad(item, (context,context), 'constant', constant_values=(0,0)))
    labels=torch.from_numpy(np.concatenate(lab))
    return labels


# In[45]:


def get_index(X):
    X = np.load(X,allow_pickle=True)
    lst = []
    for item in X:
        lst.append(np.pad(np.ones(item.shape[0]).astype(bool), (context,context), constant_values=False, mode='constant'))
    lst = np.concatenate(lst)
    index = np.array(range(len(lst)))
    use_index = lst[index].tolist()
    return use_index


# In[46]:


train = data_load('train.npy')
train_labels = load('train_labels.npy')
train_index = get_index('train_labels.npy')
dev = data_load('dev.npy')
dev_labels = load('dev_labels.npy')
dev_index = get_index('dev.npy')
test = data_load('test.npy')
test_index= get_index('test.npy')


# In[47]:


class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.Y)

    def __getitem__(self,index):
        X = self.X[index-context:index+context+1].float().reshape(-1) #flatten the input
        Y = self.Y[index].long()
        return X,Y


# In[48]:


class Dataset(data.Dataset):
    def __init__(self, X):
        self.X = X
    def __len__(self):
        return len(self.X)

    def __getitem__(self,index):
        X = self.X[index-context:index+context+1].float().reshape(-1) #flatten the input
        return X


# In[49]:


class Sampler(data.Sampler):
    def __init__(self, data_source, index, train=False):    
        super().__init__(data_source)
        self.data_source = data_source
        self.index = index
        self.train = train
        
    def __iter__(self):
        if self.train:
            np.random.shuffle(self.index)
            return iter(self.index)
        else:
            return iter(self.index)
    
    def __len__(self):
        return len(self.data_source)


# In[50]:


cuda = torch.cuda.is_available()
num_workers = 0
# Training
train_dataset = MyDataset(train, train_labels)

sampler_train = Sampler(train_dataset, train_index, train=True)

train_loader_args = dict(shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler = sampler_train) if cuda                    else dict(shuffle=True, batch_size=batch_size)
train_loader = data.DataLoader(train_dataset, **train_loader_args)

# Dev
dev_dataset = MyDataset(dev, dev_labels)

smapler_dev = Sampler(dev_dataset, dev_index, train=False)

dev_loader_args = dict(shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler = sampler_dev) if cuda                    else dict(shuffle=True, batch_size=batch_size)

dev_loader = data.DataLoader(dev_dataset, **dev_loader_args)

# Test
test_dataset = Dataset(test)

sampler_test = Sampler(test_dataset, test_index, train=False)

test_loader_args = dict(shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, sampler = sampler_test) if cuda                    else dict(shuffle=True, batch_size=batch_size)
test_loader = data.DataLoader(test_dataset, **test_loader_args)


# In[51]:


# MODEL DEFINITION
class Simple_MLP(nn.Module):
    def __init__(self):
        super(Simple_MLP, self).__init__()
        layers = []

        layers.append(nn.Linear(input_size,2048))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(2048))

        layers.append(nn.Linear(2048,4096))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(4096))
        
        layers.append(nn.Linear(4096,2048))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(2048))
        
        layers.append(nn.Linear(2048,1024))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        layers.append(nn.Linear(1024,512))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(512, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# In[53]:


#Create the model and define the Loss and Optimizer
input_size = 13* (context*2+1)
output_size = 346
model = Simple_MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
device = torch.device("cuda" if cuda else "cpu")
model.to(device)
print(model)


# In[54]:


#Create a function that will train the network for one epoch
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):   
        optimizer.zero_grad()   # .backward() accumulates gradients
        data = data.to(device)
        target = target.to(device) # all data & model on same device

        outputs = model(data)
        
        predicted = torch.max(outputs.data, 1)
        total_predictions += target.size(0)
        correct_predictions += (predicted[1] == target).sum().item()
        
        loss = criterion(outputs, target)
        running_loss += loss.item()
        if batch_idx//100 == 0:
            print('========')
        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    
    running_loss /= len(train_loader)
    acc = (correct_predictions/total_predictions)*100.0
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    print('Training Accuracy: ', acc, '%')
    return running_loss


# In[55]:


#Create a function that will evaluate our network's performance on the test set
def test_model(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (data, target) in enumerate(test_loader):   
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)

            predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted[1] == target).sum().item()

            loss = criterion(outputs, target).detach()
            running_loss += loss.item()


        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc


# In[ ]:


n_epochs = 15
Train_loss_lst = []
dev_loss_lst = []
dev_acc_lst = []
MODEL_NAME = 'model'
""" traing the model """

for i in range(n_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    dev_loss, dev_acc = test_model(model, dev_loader, criterion)
    Train_loss_lst.append(train_loss)
    dev_loss_lst.append(dev_loss)
    dev_acc_lst.append(dev_acc)
    print('='*20)
    
    # may save the training model for future use
    if not os.path.exists("./model"):
        os.mkdir("./model")

    torch.save(model.state_dict(),'/home/unbuntu/11785/model/{}_{}'.format(MODEL_NAME,i))
    logging.info('model saved to ./model/{}_{}'.format(MODEL_NAME,i))  


# In[ ]:


#Test
net.load_state_dict(torch.load('/home/unbuntu/11785/model/model_1'))


# In[ ]:


preds = []

with torch.no_grad():
    for x_batch in test_loader:
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        
        predicted = torch.max(outputs.data, 1)
        preds.append(predicted)

preds = torch.cat(preds)
preds = preds.cpu().numpy


# In[ ]:


import pandas as pd


# In[ ]:


Id = np.array(range(len(preds)))


# In[ ]:


pre = {'id': Id,
       'label': preds,
       }
df = pd.DataFrame(pre)
df.to_csv("submit_MLP.csv", index=False)


# In[ ]:


get_ipython().system('kaggle competitions submit -c 11-785-fall-20-slack-homework-1-part-2 -f submission.csv -m "Message"')

