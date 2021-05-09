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
#%%
'''
#loading the CSV file 

train_csv=pd.read_csv('fashion-mnist_train.csv')
test_csv=pd.read_csv('fashion-mnist_test.csv')
#%%

#Building the dataset class #if you go for CSV file as dataset
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
        
''Transforming the dataset to tensor''' 
"""
train_set = FashionDataset(train_csv,transform=transforms.Compose([transform.ToTensor()]))     
test_set = FashionDataset(test_csv,transform=transforms.Compose([transform.ToTensor()]))     
        
train_loader = DataLoader(train_set,batch_size=64)        
        
test_loader = DataLoader(test_set,batch_size=64)

"""

#%%

''' Downloading the FashionMNIST class from torchvision module'''

train_set = torchvision.datasets.FashionMNIST("Dataset", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("Dataset", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))  
#%%
train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=100)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=100)

def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]




#%%
print(len(train_set))
a = next(iter(train_loader))
print(a[0].size())
image, label = next(iter(train_set))
plt.imshow(image.squeeze(), cmap="gray")
print(label,output_label(label))
#%%
image, label = next(iter(train_set))
plt.imshow(image.squeeze(), cmap="gray")
print(label)
demo_loader = torch.utils.data.DataLoader(train_set, batch_size=10)

batch = next(iter(demo_loader))
images, labels = batch
print(type(images), type(labels))
print(images.shape, labels.shape)
grid = torchvision.utils.make_grid(images, nrow=10)

plt.figure(figsize=(15, 20))
plt.imshow(np.transpose(grid, (1, 2, 0)))
print("labels: ", end=" ")
for i, label in enumerate(labels):
    print(output_label(label), end=", ")

#%%
'''Building the CNN'''        
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # layer one
        self.layer1=nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,padding=1),
                                  nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer2=nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                                  nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2))
        self.fc1=nn.Linear(in_features=2300, out_features=600)
        self.drop=nn.Dropout2d(0.25)
        self.fc2=nn.Linear(in_features=600, out_features=120)
        self.fc3=nn.Linear(in_features=120, out_features=10)
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=out.view(out.size(0),-1)
        out=self.fc1(out)
        out=self.drop(out)
        out=self.fc2(out)
        out=self.fc3(out)
        return out

#%%

model=CNN()
model.to(device)
error=nn.CrossEntropyLoss()
learning_rate=0.01
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
print(model)
#%%
'''Traning the model''' 
num_epochs=6
count=0
#for visulization of loss and accuraacy
loss_list=[]
iteration_list=[]
accuracy_list=[]
#for accuracy
prediction_list=[]
labels_list=[]

for epoch in range(num_epochs):
    for images,labels in train_loader :
        images,labels=images.to(device),labels.to(device)
        train=Variable(image.view(100,1,28,28))
        labels=Variable(labels)
        
    # forward pass
    outputs=model(train)
    loss=error(outputs,labels)
    
    #initializing gradiants to zero
    optimizer.zero_grad()
    
    #propagating the error
    loss.backward()
    
    #optimizing the parameter
    
    optimizer.step()
    
    count +=1
    '''Testing the model'''
    if not (count % 50):
        total=0
        correct=0
        for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            labels_list.append(labels)
            test=Variable(images.view(100,1,28,28))
            outputs=model(test)
            predictions=torch.max(outputs,1)[1].to(device)
            prediction_list.append(predictions)
            correct += (predictions == labels).sum()
            total += len(labels)
        accuracy = correct * 100 / total
        loss_list.append(loss.data)
        iteration_list.append(count)
        accuracy_list.append(accuracy)
        
        if not (count % 500):
            print("Iteration :{},loass:{},Accuracy:{}%".format(count,loss.data,accuracy))
#%%
        
        
        
        
        
        
        