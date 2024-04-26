import torch 
import torch.nn as nn 

class TwoLayerCNN(nn.Module):
    def __init__(self, output=3):
        super(TwoLayerCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )        
        self.fc = nn.Linear(200704, output) 
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # register the hook
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.fc(x)
        return x

class FourLayerCNN(nn.Module):
    def __init__(self, output=3):
        super(FourLayerCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )    
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )            
        self.fc = nn.Linear(12544, output) # 224 

        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x










