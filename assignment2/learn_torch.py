from cProfile import label
import imp
import numpy as np
from pyparsing import col
import torch 
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda
import matplotlib.pyplot as plt
data=[[1,2,],[3,4]]
x_data=torch.tensor(data)
np_array=np.array(data)
x_np=torch.from_numpy(np_array)

train_data=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data=datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map={
    0:"T-shirt",
    1:"Trouser",
    2:"Pullover",
    3:"Dress",
    4:"Coat",
    5:"Sandal",
    6:"Shirt",
    7:"Sneaker",
    8:"Bag",
    9:"Ankle Boot",
}

figure=plt.figure(figsize=(8,8))
cols,rows=3,3
for i in range(1,cols*rows+1):
    sample_idx=torch.randint(len(train_data),size=(1,)).item()
    img,label=train_data[sample_idx]
    figure.add_subplot(rows,cols,i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(),cmap="gray")
plt.show()

#x_ones=torch.ones_like(x_data)
#x_rand=torch.rand_like(x_data,dtype=torch.float)
'''
tensor=torch.rand(3,4)
print(tensor.device)#tensor.shape ,tensor.dtype
if torch.cuda.is_available():
    tensor=tensor.to('cuda')
print(tensor.device)
'''
tensor=torch.ones(4,4)
print(tensor[0])
print(tensor[:,-1])

y1=tensor@tensor.T
y2=tensor.matmul(tensor.T)
y3=torch.rand_like(tensor)
torch.matmul(tensor,tensor.T,out=y3)

z1=tensor*tensor
z2=tensor.mul(tensor)


