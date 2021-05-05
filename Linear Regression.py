# -*- coding: utf-8 -*-
"""
Created on Wed May  5 22:35:56 2021

Linear Regression using pytorch 

@author: smith_barbose @neb.tics
"""
'''
Step 1) Design model (input,output_size,forward_pass)
Step 2) Construct loss and optimizer 
Step 3) Traning Loop:
    -Forward pass: compute prediction & loss
    -Backward pass: grradients
    -Update weights 
'''

#importing libraries
import torch 
import torch.nn as nn
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt

# prepraing dataset 

X_numpy,y_numpy=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)

# converting it to tensor form
X= torch.from_numpy(X_numpy.astype(np.float32))
y= torch.from_numpy(y_numpy.astype(np.float32))

#Reshaping the y
y=y.view(y.shape[0],1)

'''Step 1 model building '''

n_samples,n_features=X.shape

in_features=n_features
out_features=1

model= nn.Linear(in_features, out_features)

'''   Step 2   Loss and optimizer '''

learning_rate=0.01
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)

''' Step 3  Traning loop'''

num_epoch=100
for epoch in range(num_epoch):
    #forward passs & loss
    y_predicted=model(X)
    loss=criterion(y_predicted,y)
    
    #backward
    loss.backward()
    optimizer.step()

    
    #update 
    optimizer.zero_grad()
    #printing
    if (epoch+1)%10==0:
        print(f'epoch:{epoch+1},loss={loss.item():.3f}')
#ploting
predicted=model(X).detach().numpy()#detech the operation 
plt.plot(X_numpy,y_numpy,'ro')
plt.plot(X_numpy,predicted,'b')
plt.show()