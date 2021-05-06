# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:31:35 2021

The concept of Dataset & Dataloader  using pytorch

@author: smith_barbose @neb.tics

In this code we learn the basic of importing data & 

Epoch -> 1 forward and backward pass of all traning samples

batch_size -> number of training samples in one fprward & backward pass

number_of_iteration -> number of pass , each pass using [batch_size]

eg --> 100 samples,batch_size=20 --> 100/20 = 5 for iteration of 1 epoch  


"""

import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math

'''Using the wine dataset '''

class wine(Dataset):
    def __init__(self):
        xy=np.loadtxt('Dataset/Wine.csv',delimiter=',',dtype=np.float32,skiprows=1)
        self.n_samples=xy.shape[0]
        self.x_data=torch.from_numpy(xy[:,1:])
        self.y_data=torch.from_numpy(xy[:,[0]])
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.n_samples 

dataset=wine()
first_data=dataset[0]
features,lables=first_data
print(features,lables)

train_loader=DataLoader(dataset=dataset,batch_size=4,shuffle=True)

#converting to an iterator 

dataiter=iter(train_loader)
data=dataiter.next()
features,lables=data
print("------------------------------------------------")
print(f'{features},{lables}')
        
#dummy traning
num_epochs=2
total_samples=len(dataset)
n_iteration=math.ceil(total_samples/4)#dividing by the batch_size
print(total_samples,n_iteration)
for epoch in range(num_epochs):
    for i,(input,lables) in enumerate(train_loader):
        if (i+1)%5==0:
            print(f'Epoch:{epoch+1}/{num_epochs},Steps{i+1}/{n_iteration} | input{input.shape}|lables{lables.shape}')
print('Compleated')
        