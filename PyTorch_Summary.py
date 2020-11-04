# The structure of training a classification model 
# prepare train and test data: features and label (such as images or texts)
# specify the structure of a neural network model
# train a model
# performance metrics

# Reference Udacity - Deep Learning With PyTorch

import torch
from torchvision import datasets, transforms

# tensor is a main element to be used for calculation in torch
# A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
# https://pytorch.org/docs/stable/tensors.html
ta = torch.from_numpy(np.array([[1,2,3],[7,8,9]]))  
index = torch.tensor([[1,2,1],[0,1,0]])
ta.gather(1, index) # get some items from tensor matrix
ta.unsqueeze(1) # add an additional dimension

# use tensor to calculate gradients
tx = torch.randn(10, requires_grad=True)
y = tx.mean() # y is a scalor
y.backward() # calculate gradients
tx.grad # gradients dy/d(tx) 
