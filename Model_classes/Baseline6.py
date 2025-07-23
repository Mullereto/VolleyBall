import torch.nn as nn
import torchvision.models as models

class featuregeter(nn.Module):
    def __init__(self, num_class = 9):
        super(featuregeter, self).__init__()
        
        self.resnet = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=self.resnet.fc.in_features, out_features=num_class)
        )
    
    def forward(self, x):
        return self.resnet(x)
        


class Baseline6(nn.Module):
    def __init__(self, featuregeter, hidden_dim = 1024, num_class = 8):
        super(Baseline6, self).__init__()
        
        self.featuregeter = nn.Sequential(*list(featuregeter.resnet.children())[:-1])
        
        for param in self.featuregeter.parameters():
            param.requires_grad = False
        
        self.maxpool = nn.AdaptiveMaxPool2d((1,2048))
        
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
        B, T, P, C, H, W = x.shape #Batch, sequnce, number_players, channls, hight, width
        x = x.view(B*T*P, C, H, W) 
        x = self.featuregeter(x) #[B*T*P, 2048,1,1]
        
        x = x.view(B*T, P, -1)
        x = self.maxpool(x) #[B*T, 1, 2048]
        x = x.squeeze(dim=1)
        
        x = x.view(B, T, 2048)
        
        x, (h_n, c_n) = self.lstm(x)
        
        x = x[:,-1,:] #last hidden state
        
        x = self.the_classifier(x)
        return x