import imp
from catboost import train
from matplotlib import image
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torch import nn 
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
from cs231n.net_visualization_pytorch import make_fooling_image
# Download and load the pretrained SqueezeNet model.
model = torchvision.models.squeezenet1_1(pretrained=True)

# We don't want to train the model, so tell PyTorch not to compute gradients
# with respect to model parameters.
for param in model.parameters():
    param.requires_grad = False


toPIL=transforms.ToPILImage()

toTensor=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224))   
])
input_image=Image.open('Goldfish.jpg')

input_tensor=toTensor(input_image)

print(input_tensor.shape)
transforms=transforms.Compose([
    transforms.ToTensor(),
])
train_data=datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=transforms,
)

subset_train_data=Subset(train_data,indices=[1])
img,y=subset_train_data.__getitem__(0)
print(img.shape)

#input_tensor_permute=torch.permute(input_tensor,(1,2,0))
#plt.imshow(input_tensor_permute)
X_fool=make_fooling_image(torch.unsqueeze(input_tensor,0),target_y=147,model=model)

X_fool_squeeze=torch.squeeze(X_fool,0)
print(X_fool_squeeze.shape)
pic=toPIL(X_fool_squeeze)

X_fool_squeeze_permute=torch.permute(X_fool_squeeze,(1,2,0))
plt.imshow(X_fool_squeeze_permute.detach().numpy())
plt.show()
pic.save('ramdom.jpg')
