import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torch.autograd import Variable
from PIL import Image
from oodcl import OodCls

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
    transforms.RandomCrop((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])



od=OodCls()

mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

images, labels = next(iter(mnist_testloader))
out=od.classify(images)
print(labels)
print(out)



cifar10_testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
cifar10_testloader = DataLoader(cifar10_testset, batch_size=64, shuffle=False)

images, labels = next(iter(cifar10_testloader))
out=od.classify(images)
print(out)
