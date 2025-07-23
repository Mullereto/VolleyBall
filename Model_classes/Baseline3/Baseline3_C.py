import torch
from torch import nn

class NNClassifier(nn.Module):
    def __init__(self):
        super(NNClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 8),
        )

    def forward(self, x):
        # # Max pool over players (12)
        # x, _ = torch.max(x, dim=2)
        # # Average pool over frames (9)
        # x = torch.mean(x, dim=1)
        # # Flatten to get feature vector
        # x= x.view(x.size(0), -1)
        # # Pass through the fully connected layer
        x = self.fc(x)
        return x

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class NNClassifier(nn.Module):
#     def __init__(self, num_classes=8):
#         super(NNClassifier, self).__init__()
        
#         # Convolutional layers to extract spatial features
#         self.conv1 = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling

#         # Fully connected layers
#         self.fc1 = nn.Linear(128, 64)
#         self.fc2 = nn.Linear(64, num_classes)
        
#     def forward(self, x):
#         # Reshape to (batch, channels, height, width)
#         x = x.permute(0, 3, 1, 2)  # From (batch, 9, 12, 2048) -> (batch, 2048, 9, 12)
        
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
        
#         x = self.pool(x)  # Global average pooling
#         x = torch.flatten(x, start_dim=1)  # Flatten
        
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)  # Output logits

#         return x
