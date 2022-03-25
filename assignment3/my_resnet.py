from matplotlib import image
from more_itertools import sample
from sklearn.svm import LinearSVR
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset #CIFAR   etc 
import torchvision.transforms as T

import numpy as np

import torch.nn.functional as F  # useful stateless functions

writer=SummaryWriter()

USE_GPU = True
dtype = torch.float32 # We will be using float throughout this tutorial.

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss.
print_every = 100
print('using device:', device)

NUM_TRAIN = 49000

# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've #!hardcoded the mean and std.
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))#for batch ramdon init

cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True, 
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)

for X,y in loader_train:
    print('Shape of X[N,C,W,H]:',X.shape)
    print('Shape of y:',y.shape)
    break

def check_accuracy(loader, model,temp):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    loss_avg=0.0
    model.eval()  # set model to evaluation mode#!
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            loss_avg+=loss.item()
        acc = float(num_correct) / num_samples
        #writer.add_scalars('val_loss',loss_avg,temp)
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


def train(model, optimizer, epochs=10):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    temp=0
    loss_avg=0.0
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()
            
            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()
            
            loss_avg+=loss.item()
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss_avg))
                writer.add_scalar('train_loss',loss_avg,temp)
                temp+=1
                check_accuracy(loader_val, model,temp)
                loss_avg=0.0


class Residualblock(nn.Module):
    def __init__(self,in_channel,out_channel) -> None:
        super().__init__()
        self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1)
        nn.init.kaiming_normal(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        
        self.bn1=nn.BatchNorm2d(out_channel)
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        
        self.conv2=nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1)
        nn.init.kaiming_normal(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        self.bn2=nn.BatchNorm2d(out_channel)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn1.bias)
        
        self.relu=nn.ReLU()
        
    def forward(self,x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out+=x
        out=self.relu(out)
        return (out)    

class Flatten(nn.Module):
    def forward(self, x):
        N = x.shape[0] # read in N, C, H, W
        return x.view(N, -1)
    
model = nn.Sequential(
    nn.Conv2d(3,64,kernel_size=3,padding=1),
    Residualblock(64,64),
    Residualblock(64,64),
    nn.Conv2d(64,128,kernel_size=2,stride=2,),
    Residualblock(128,128),
    Residualblock(128,128),
    nn.Conv2d(128,256,kernel_size=2,stride=2),
    Residualblock(256,256),
    Residualblock(256,256),
    nn.Conv2d(256,512,kernel_size=2,stride=2),
    Residualblock(512,512),
    Residualblock(512,512),
    nn.MaxPool2d(kernel_size=4),
    Flatten(),
    nn.Linear(512,10),
)

optimizer = optim.Adam(model.parameters())
images,labels=next(iter(loader_train))
writer.add_graph(model,images)

train(model, optimizer)

best_model = model
check_accuracy(loader_test, best_model)
