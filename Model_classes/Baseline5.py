import torch
import torch.nn as nn
from torchvision import models

class Baseline5_feature_extractor_per_person(nn.Module):
    def __init__(self, hidden_dim = 1024, num_class = 9):
        super(Baseline5_feature_extractor_per_person, self).__init__()
        self.resnet = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).children())[:-1]
        )
        
        self.hidden_dim = hidden_dim
        
        
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        
        self.the_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        print("###################in the forward###############")
        print(x.shape)
        B, T, P, C, H, W = x.size() # (16, 9, 12, 3, 224, 224)
        
        x = x.view(B * T * P, C, H, W)  # Flatten all crops
        features = self.resnet(x)  # (B * T * P, feature_dim, 1, 1)

        # Now, run LSTM for each player across time
        features = features.view(B * P, T, -1)
        lstm_out, _ = self.lstm(features)  # (B * P, T, hidden_dim)
        last_hidden = lstm_out[:, -1, :]   # (B * P, hidden_dim)

        # Final classification
        out = self.the_classifier(last_hidden)  # (B, num_class)
        out = out.view(B, P, -1)             # (B, P, num_class)
        out = out.mean(dim=1)    

        return out


class for_group(nn.Module):
    def __init__(self, feature_extractor_per_person, num_class=8):
        super(for_group, self).__init__()
        self.resnet = feature_extractor_per_person.resnet
        self.lstm = feature_extractor_per_person.lstm
        self.hidden = feature_extractor_per_person.hidden_dim
        
        for para in self.resnet.parameters():
            para.requires_grad = False
        
        for para in self.lstm.parameters():
            para.requires_grad = False
            
        self.max_pool = nn.AdaptiveMaxPool2d((1,2048))
        
        self.fc = nn.Sequential(
            nn.Linear(2048 , 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_class),
        )
    def forward(self, x):
        B, T, P, C, H, W = x.size() # (16, 9, 12, 3, 224, 224)
        
        x = x.view(B * T * P, C, H, W)  # Flatten all crops
        features = self.resnet(x)  # (B * P * T, feature_dim, 1, 1)
        features = features.view(B * P, T, -1)
        movement, _ = self.lstm(features)
        
        x = torch.cat([features, movement], dim=2)
        x = x.contiguous()
        
        x = x[:,-1,:]
        x = x.view(B, P, -1)
        x = self.max_pool(x)
        x = x.squeeze(dim=1)
        
        x = self.fc(x)
        
        return x
        