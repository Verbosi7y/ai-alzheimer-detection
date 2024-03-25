'''
    AlzheimerModel.py -- Custom PyTorch Alzheimer CNN model.
    Authors: Darwin Xue
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class AlzheimerCNN(nn.Module):
    def __init__(self, input_size=1):
        super(AlzheimerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=16)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=16)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(input_size, 50) # Fully Connected Layer 1
        self.fc2 = nn.Linear(50, 4) # Fully Connected Layer 2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x