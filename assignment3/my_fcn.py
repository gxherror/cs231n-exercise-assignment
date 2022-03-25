import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.models as models
import torchvision.datasets as dset #CIFAR   etc 
import torchvision.transforms as T
from PIL import Image
import numpy as np

USE_GPU = True
dtype = torch.float32 # We will be using float throughout this tutorial.

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss.
print_every = 100
print('using device:', device)


# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
input_image = input_image.convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# 定义双线性插值，作为转置卷积的初始化权重参数
def bilinear_kernel(in_channels, out_channels, kernel_size):
   factor = (kernel_size + 1) // 2
   if kernel_size % 2 == 1:
       center = factor - 1
   else:
       center = factor - 0.5
   og = np.ogrid[:kernel_size, :kernel_size]
   filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
   weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
   weight[range(in_channels), range(out_channels), :, :] = filt
   return torch.from_numpy(weight)


class FCN(nn.Module):
   def __init__(self, num_classes):
       super(FCN, self).__init__()
       pretrained_net = models.resnet34(pretrained=True)
       self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4]) # 第一段
       self.stage2 = list(pretrained_net.children())[-4] # 第二段
       self.stage3 = list(pretrained_net.children())[-3] # 第三段
       
       # 通道统一
       self.scores1 = nn.Conv2d(512, num_classes, 1)
       self.scores2 = nn.Conv2d(256, num_classes, 1)
       self.scores3 = nn.Conv2d(128, num_classes, 1)
       
       # 8倍上采样
       self.upsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)
       self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16) # 使用双线性 kernel
       
       # 2倍上采样
       self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
       self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4) # 使用双线性 kernel
       self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)   
       self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4) # 使用双线性 kernel

       
   def forward(self, x):
       x = self.stage1(x)
       s1 = x # 1/8
       
       x = self.stage2(x)
       s2 = x # 1/16
       
       x = self.stage3(x)
       s3 = x # 1/32
       
       s3 = self.scores1(s3)
       s3 = self.upsample_2x(s3) # 1/16
       s2 = self.scores2(s2)
       s2 = s2 + s3
       
       s1 = self.scores3(s1)
       s2 = self.upsample_4x(s2) # 1/8
       s = s1 + s2

       s = self.upsample_8x(s) # 1/1
       return s
   
x = torch.randn(1,3,64,64) # 伪造图像
num_calsses = 21
net = FCN(num_calsses)
y = net(x)
print(y.shape)

# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

import matplotlib.pyplot as plt
plt.imshow(r)
# plt.show()