import torch
import torch.nn as nn
from torchvision import models

class Baseline4(nn.Module):
    def __init__(self, model_path, hidden_dim = 1024, num_class = 8):
        super(Baseline4, self).__init__()
        self.device = torch.device('cuda')
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features
        
        self.hidden_dim = hidden_dim
        
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        
        self.the_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(128, num_class)
        )

    def forward(self, x):
        B, T, C, H, W = x.size() # (16, 9, 3, 224, 224)
        
        x = x.view(B * T, C, H, W).to(self.device) #(16*9, 3, 224, 224)
        
        features = self.feature_extractor(x)  #(16*9, 3, 1, 1)
        features = features.view(B, T, -1)
        
        lstm_out, _ = self.lstm(features)           # [B, T, hidden_dim]
        out = self.the_classifier(lstm_out[:, -1, :])   # take last step for classification
        
        return out
        