# import deep learning
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class DefConvNetBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, 
                 kernel_size, defconv=False):
        
        super().__init__()
        
        # Creating Deformable Convolutional Network
        if defconv: 
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels, 
                          kernel_size=kernel_size),
                deform_conv2d(),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels, 
                          kernel_size=kernel_size),
                nn.BatchNorm2d(out_channels),
            )
    def forward(self, x):
        x = self.block(x)
        return x
    


class DCNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.module1 = DefConvNetBlock(1, 32, kernel_size=5)
        self.module2 = DefConvNetBlock(32, 64, kernel_size=3)
        self.module3 = DefConvNetBlock(64, 128, kernel_size=3)
        self.module4 = DefConvNetBlock(128, 128, kernel_size=3)
        
        self.fc1 = nn.Linear(42*42*128, 128)
        self.fc2 = nn.Linear(128, 8)
    
    def forward(self, x):
        
        x = self.module1(x)
        x = F.relu(x)
        
        x = self.module2(x)
        x = F.relu(x)
        
        x = self.module3(x)
        x = F.relu(x)
        
        x = self.module4(x)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        
        # creating output
        output = torch.sigmoid(x)
        return output


