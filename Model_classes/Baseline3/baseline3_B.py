import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.FeatureExtractor = nn.Sequential(*list(self.model.children())[:-1])
        
    def forward(self, x):
        x = self.FeatureExtractor(x)
        return x.view(x.size(0), -1)