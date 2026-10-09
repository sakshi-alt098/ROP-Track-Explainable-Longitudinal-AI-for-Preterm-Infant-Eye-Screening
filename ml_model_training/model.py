"""
Module 3: Custom Deep Convolutional Neural Network Architecture (98.48% Baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ROPCustomCNN(nn.Module):
    """
    Exact deep CNN block architecture matching the 98.48% study:
    Conv2D(32) -> Conv2D(64) -> Conv2D(64) -> Conv2D(64) -> Dense(64) -> Dense(10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)

        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 14 * 14, 64)
        self.fc2   = nn.Linear(64, num_classes)
        self.drop  = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

    def get_cam_target_layer(self):
        return self.conv4
