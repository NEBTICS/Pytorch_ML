#!/usr/bin/env python
# coding: utf-8

# ### This is the tutorial of deep learning on FashionMNIST dataset using Pytorch. We will build a Convolutional Neural Network for predicting the classes of Dataset. I am assuming you know the basics of deep leanrning like layer architecture... convolution concepts. Without further ado... Lets start the tutorial.

# # **Importing Important Libraries**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix


# ### If the GPU is available use it for the computation otherwise use the CPU.

# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# There are 2 ways to load the Fashion MNIST dataset. 
# 
# 
#     1.   Load csv and then inherite Pytorch Dataset class .
#     2.   Use Pytorch module torchvision.datasets. It has many popular datasets like MNIST, FashionMNIST, CIFAR10 e.t.c.
#     
#     
# 
# *   We use DataLoader class from torch.utils.data to load data in batches  in both method.
# * Comment out the code of a method which you are not using. 
# 
# 
# 
# 
# 

# ### 1.    Using a Dataset class.
#     
#    *   First load the data from the disk using pandas read_csv() method.
# 
#    *   Now inherit Dataset class in your own class that you are building,    lets say FashionData.
# 
#         *  It has 2 methods: __get_item__( ) and __len__().
#         * __get_item__( ) return the images and labels and __len__( ) returns the number of items in a dataset.

# In[3]:


train_csv = pd.read_csv("../input/fashion-mnist_train.csv")
test_csv = pd.read_csv("../input/fashion-mnist_test.csv")


# In[4]:


class FashionDataset(Dataset):
    """User defined class to build a datset using Pytorch class Dataset."""
    
    def __init__(self, data, transform = None):
        """Method to initilaize variables.""" 
        self.fashion_MNIST = list(data.values)
        self.transform = transform
        
        label = []
        image = []
        
        for i in self.fashion_MNIST:
             # first column is of labels.
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        # Dimension of Images = 28 * 28 * 1. where height = width = 28 and color_channels = 1.
        self.images = np.asarray(image).reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]
        
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)


# In[5]:


# Transform data into Tensor that has a range from 0 to 1
train_set = FashionDataset(train_csv, transform=transforms.Compose([transforms.ToTensor()]))
test_set = FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))

train_loader = DataLoader(train_set, batch_size=100)
test_loader = DataLoader(train_set, batch_size=100)


# ### 2. Using FashionMNIST class from torchvision module.
# 
# 
# *   It will download the dataset first time.
# 
# 
# 

# In[6]:


train_set = torchvision.datasets.FashionMNIST(r"C:\Users\s\Desktop\neb.tics\Pytorch\Pytorch_ML\Dataset\Dataset\FashionMNIST", download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST(r"C:\Users\s\Desktop\neb.tics\Pytorch\Pytorch_ML\Dataset\Dataset\FashionMNIST", download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()]))  
                                               


# In[7]:



train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=100)
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=100)
                                          


# ### We have 10 types of clothes in FashionMNIST dataset.
# 
# 
# > Making a method that return the name of class for the label number.
# ex. if the label is 5, we return Sandal.
# 
# 

# In[8]:


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


# ### Playing with data and displaying some images using matplotlib imshow() method.
# 
# 
# 
# 

# In[9]:


a = next(iter(train_loader))
a[0].size()


# In[10]:


len(train_set)


# In[11]:


image, label = next(iter(train_set))
plt.imshow(image.squeeze(), cmap="gray")
print(label)


# In[12]:


demo_loader = torch.utils.data.DataLoader(train_set, batch_size=10)

batch = next(iter(demo_loader))
images, labels = batch
print(type(images), type(labels))
print(images.shape, labels.shape)


# In[13]:


grid = torchvision.utils.make_grid(images, nrow=10)

plt.figure(figsize=(15, 20))
plt.imshow(np.transpose(grid, (1, 2, 0)))
print("labels: ", end=" ")
for i, label in enumerate(labels):
    print(output_label(label), end=", ")


# ## Building a CNN 
# 
# 
# *   Make a model class (FashionCNN in our case)
#     * It inherit nn.Module class that is a super class for all the neural networks in Pytorch.
# * Our Neural Net has following layers:
#     * Two Sequential layers each consists of following layers-
#         * Convolution layer that has kernel size of 3 * 3, padding = 1 (zero_padding) in 1st layer and padding = 0 in second one. Stride of 1 in both layer.
#         * Batch Normalization layer.
#         * Acitvation function: ReLU.
#         * Max Pooling layer with kernel size of 2 * 2 and stride 2.
#      * Flatten out the output for dense layer(a.k.a. fully connected layer).
#      * 3 Fully connected layer  with different in/out features.
#      * 1 Dropout layer that has class probability p = 0.25.
#   
#      * All the functionaltiy is given in forward method that defines the forward pass of CNN.
#      * Our input image is changing in a following way:
#         * First Convulation layer : input: 28 \* 28 \* 3, output: 28 \* 28 \* 32
#         * First Max Pooling layer : input: 28 \* 28 \* 32, output: 14 \* 14 \* 32
#         * Second Conv layer : input : 14 \* 14 \* 32, output: 12 \* 12 \* 64
#         * Second Max Pooling layer : 12 \* 12 \* 64, output:  6 \* 6 \* 64
#     * Final fully connected layer has 10 output features for 10 types of clothes.
# 
# > Lets implementing the network...
# 
# 
# 
# 

# In[14]:


class FashionCNN(nn.Module):
    
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out


# ### Making a model of our CNN class
# 
# *   Creating a object(model in the code)
# *   Transfering it into GPU if available.
# *  Defining a Loss function. we're using CrossEntropyLoss() here.
# *  Using Adam algorithm for optimization purpose.
# 
# 

# In[15]:


model = FashionCNN()
model.to(device)

error = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)


# ## Training a network and Testing it on test dataset

# In[16]:


num_epochs = 5
count = 0
# Lists for visualization of loss and accuracy 
loss_list = []
iteration_list = []
accuracy_list = []

# Lists for knowing classwise accuracy
predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Transfering images and labels to GPU if available
        images, labels = images.to(device), labels.to(device)
    
        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)
        
        # Forward pass 
        outputs = model(train)
        loss = error(outputs, labels)
        
        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()
        
        #Propagating the error backward
        loss.backward()
        
        # Optimizing the parameters
        optimizer.step()
    
        count += 1
    
    # Testing the model
    
        if not (count % 50):    # It's same as "if count % 50 == 0"
            total = 0
            correct = 0
        
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels_list.append(labels)
            
                test = Variable(images.view(100, 1, 28, 28))
            
                outputs = model(test)
            
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
            
                total += len(labels)
            
            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        
        if not (count % 10):
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))


# ### Visualizing the Loss and Accuracy with Iterations
# 
#%%
iteration_list=torch.FloatTensor(iteration_list)
iteration_list=iteration_list.cpu()
loss_list=torch.FloatTensor(loss_list)
loss_list=loss_list.cpu()
accuracy_list=torch.FloatTensor(accuracy_list)
accuracy_list=accuracy_list.cpu()
# In[17]:
print(a)

print(accuracy_list.detach().numpy())

#%%
plt.plot(a, l)
plt.xlabel("No. of Iteration")
plt.ylabel("Loss")
plt.title("Iterations vs Loss")
plt.show()


# In[18]:


plt.plot(iteration_list, accuracy_list)
plt.xlabel("No. of Iteration")
plt.ylabel("Accuracy")
plt.title("Iterations vs Accuracy")
plt.show()


# ### Looking the Accuracy in each class of FashionMNIST dataset

# In[19]:


class_correct = [0. for _ in range(10)]
total_correct = [0. for _ in range(10)]

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        test = Variable(images)
        outputs = model(test)
        predicted = torch.max(outputs, 1)[1]
        c = (predicted == labels).squeeze()
        
        for i in range(100):
            label = labels[i]
            class_correct[label] += c[i].item()
            total_correct[label] += 1
        
for i in range(10):
    print("Accuracy of {}: {:.2f}%".format(output_label(i), class_correct[i] * 100 / total_correct[i]))


# ### Printing the Confusion Matrix 

# In[20]:


from itertools import chain 

predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
predictions_l = list(chain.from_iterable(predictions_l))
labels_l = list(chain.from_iterable(labels_l))


# In[21]:


import sklearn.metrics as metrics

confusion_matrix(labels_l, predictions_l)
print("Classification report for CNN :\n%s\n"
      % (metrics.classification_report(labels_l, predictions_l)))


# ### This is my implementation of deep learning in FashionMNIST dataset using Pytorch. I've achieved 93% test accuracy. Change those layer architecture or parameters to make it better. 
# ***I hope you like it. Give your feedback. It helps me to a lot. Thank you. :)***
