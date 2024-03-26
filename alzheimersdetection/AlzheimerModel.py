'''
    AlzheimerModel.py -- Custom PyTorch Alzheimer CNN model.
    Authors: Darwin Xue
'''
import torch.nn as nn
import torch.nn.functional as F


class AlzheimerCNN(nn.Module):
    def __init__(self, input_size=1):
        super(AlzheimerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=32 * 30 * 30, out_features=64) # Fully Connected Layer 1
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 4) # Fully Connected Layer 2

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(-1, 32 * 30 * 30)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        return x