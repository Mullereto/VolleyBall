import torch.nn as nn
import torch
from torchvision import models
import matplotlib.pyplot as plt

class NNClassifier(nn.Module):
    def __init__(self):
        super(NNClassifier, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d((1, 2048))  # Output shape: (batch_size, 9, 2048)
        #self.avg_pool = nn.AdaptiveAvgPool2d((1, 2048))  # Output shape: (batch_size, 9, 2048)
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 8)
        )
        
    def forward(self, x):
        print("Before pooling: ", x.shape)  # (batch_size, 9, 12, 2048)
        
        x = self.max_pool(x).squeeze()  # (batch_size, 9, 2048)
        
        print("After pooling: x_max: ", x.shape)
        
        x = torch.mean(x, dim=1)  # Average over frames → (batch_size, 2048)
        
        print("After mean pooling over frames: ", x.shape)
        
        x = self.fc(x)
        return x