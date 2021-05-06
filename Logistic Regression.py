# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:00:05 2021

Logistic Regression using pytorch 

@author: smith_barbose @neb.tics
"""
'''
Using the breast cancer dataset from the sklearn moduel
first we need to transform and preprocess the data 
then 

Step 1) Preparing the dataset
Step 2) Design model (input,output_size,forward_pass)
Step 3) Construct loss and optimizer 
Step 4) Traning Loop:
    -Forward pass: compute prediction & loss
    -Backward pass: grradients
    -Update weights 
'''

'''Step 1 Preparing the dataset'''
#importing libraries
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 

# preparing data
dataset=datasets.load_breast_cancer()
X,y=dataset.data,dataset.target

n_samples,n_features=X.shape

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)

#scalar form
sc=StandardScaler()
X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)

#converting to tensor 

X_train=torch.from_numpy(X_train.astype(np.float32))
X_test=torch.from_numpy(X_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))

#reshaping the y tesnor 
y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)

'''Step 2 mdoel building '''

class LogisticeRegression(nn.Module):
    def __init__(self,n_input_feature):
        super(LogisticeRegression, self).__init__()
        self.Linear=nn.Linear(n_input_feature,1 )
    #forward pass
    def forward(self,x):
        y_predicted=torch.sigmoid(self.Linear(x))
        return y_predicted
model = LogisticeRegression(n_features)

''' Step 3 Loss and optimizer'''
criterion=nn.BCELoss()#binary cross entropy loss (BCE)
learinig_rate=0.01
optimizer= torch.optim.SGD(model.parameters(), lr=learinig_rate)  

'''Step 4 Traning loop'''     
num_epoch = 100
for epoch in range(num_epoch):
    #forward pass
    y_predicted=model(X_train)
    loss=criterion(y_predicted,y_train)
    
    #backward
    loss.backward()
    
    #update
    optimizer.step()
    
    #zero gradients
    optimizer.zero_grad()
    
    if (epoch+1)%10==0:
        print(f'epoch:{epoch+1},loss={loss.item():.4f}')
with torch.no_grad():
    y_predicted=model=model(X_test)
    y_predicted_cls=y_predicted.round()
    acc=y_predicted_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f'accuracy={acc:.4f}')
