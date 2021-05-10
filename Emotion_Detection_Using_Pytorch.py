# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:02:09 2021

@author: smith_barbose @neb.tics

download the dataset from https://www.kaggle.com/dataset/de270025c781ba47a3a6d774a0d670452bfb4dc9d2d6b13740cdb0c17aa7bf2b

"""


import os
import torch
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Looking into the directory
data_dir = r'C:\Users\s\Documents\Nebtics\Datasets\Emotions\fer2013'
print(os.listdir(data_dir))
classes_train = os.listdir(data_dir + "/train")
classes_valid = os.listdir(data_dir + "/validation")
print(f'Train Classes - {classes_train}')
print(f'Validation Classes - {classes_valid}')

'''Data transforms (Gray scaling and data augmentation'''
train_trams=tt.Compose([tt.Grayscale(num_output_channels=1),tt.RandomCrop(48,padding=4,padding_mode='reflect'),
                        tt.RandomHorizontalFlip(),tt.ToTensor(),tt.Normalize((0.5),(0.5),inplace=True)])
#test
test_trams=tt.Compose([tt.Grayscale(num_output_channels=1),
                        tt.ToTensor(),tt.Normalize((0.5),(0.5),inplace=True)])

'''Emotion detection datasets loading'''
train_ds=ImageFolder(data_dir+'/train',train_trams)
test_ds=ImageFolder(data_dir+'/validation',test_trams)

batch_size=400

'''Creating the traning and test set'''

train_dataloader=DataLoader(train_ds,batch_size,shuffle=True,num_workers=0,pin_memory=True)
test_dataloader=DataLoader(test_ds,batch_size*2,num_workers=0,pin_memory=True)

def show_batch(dl):
    for images,labels in dl:
        fig,ax=plt.subplots(figsize=(12,12))
        ax.set_xticks([]);ax.set_yticks([])
        ax.imshow(make_grid(images[:64],nrow=8).permute(1,2,0))
        break

#%%
show_batch(train_dataloader)





























