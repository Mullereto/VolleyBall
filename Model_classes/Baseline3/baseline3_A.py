import torch.nn as nn
from torchvision import models

class Baseline3(nn.Module):
    def __init__(self, num_class=9  ):
        super(Baseline3, self).__init__()
        
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        
        num_features = self.model.fc.in_features
        
        self.model.fc = nn.Linear(num_features, num_class)
        
    def forward(self, x):
        return self.model(x)