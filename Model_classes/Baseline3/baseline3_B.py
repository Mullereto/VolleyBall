import torch.nn as nn
import torch
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self, model_path):
        super(FeatureExtractor, self).__init__()
        
        self.device = torch.device('cuda')
        
        self.model = models.resnet50(weights=None)
        
        self.model.fc = nn.Linear(in_features=2048, out_features=9)
        
        trained_model = torch.load(model_path, map_location=self.device)#need to try this 
        
        self.model.load_state_dict(trained_model, strict=False)
        
        self.features = nn.Sequential(*list(self.model.children())[:-1])
        
        for param in self.features.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)