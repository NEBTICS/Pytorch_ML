# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:52:17 2021

@author: smith_barbose @neb.tics

Impimenting CNN model in pytoch 

"""
#importing libraries
import torch
import torch.nn as nn 
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#If the GPU is available the device is set to GPU else CPU
device = torch.device("cuda:0"if torch.cuda.is_available()else "cpu")
print(device)

#loading the CSV file 
train_csv=pd.read_csv('fashion-mnist_train.csv')
test_csv=pd.read_csv('fashion-mnist_test.csv')

#%%
#Building the dataset class
class FashionDataset(Dataset):
    def __init__(self,data,transform=None):
        self.fashion_MNIST=list(data.values)
        self.transform=transform
        
        label=[]
        image=[]
        
        for i in self.fashion_MNIST:
            label.append(i[0])
            image.append(i[1:])
        self.labels=np.asarray(label)
        #Dimensions of Image = 28*28*1 
        self.images=np.asarray(image).reshape(-1,28,28,1).astype('float32')
        #-1 means that the length in that dimension is inferred. 
        
    def __getitem__(self, index):
        label=self.labels[index]
        image=self.images[index]
             
        if self.transform is not None:
            image=self.transform(image)
            
        return image,label
        
    def __len__(self):
        return len(self.images)
        
'''Transforming the dataset to tensor ''' 
train_set = FashionDataset(train_csv,transform=transforms.Compose([transform.ToTensor()]))     
test_set = FashionDataset(test_csv,transform=transforms.Compose([transform.ToTensor()]))     
        
train_loader = DataLoader(train_set,batch_size=64)        
        
test_loader = DataLoader(test_set,batch_size=64)
#%%
print(len(train_set))
a = next(iter(train_loader))
print(a[0].size())
image, label = next(iter(train_set))
plt.imshow(image.squeeze(), cmap="gray")
print(label)
#%%
'''Building the CNN'''        
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        