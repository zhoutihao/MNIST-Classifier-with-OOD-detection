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



class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 11)
        
    def forward(self, x):
        x = x.view((-1, 784))
        h = self.fc1(x)
        h = F.relu(h)
        
        h = self.fc2(h)
        h = F.relu(h)
        
        h = self.fc3(h)
        out = F.softmax(h,1)
        return out

class OodCls:    
    def __init__(self, model_path='model.pt'):

        self.model = Classifier()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        self.CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    
    def classify(self, testloader):


        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
            transforms.RandomCrop((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])    
        
        images=testloader
        images = Variable(images)
        #images = self.transform(images)
        images = images.unsqueeze(0)

        outputs = self.model(images)
        _, predicted = torch.max(outputs.data, 1)

        return predicted
    

