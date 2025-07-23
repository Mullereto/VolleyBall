import torch.nn as nn
import torchvision.models as models
import torch

class EndModel(nn.Module):
    def __init__(self, person_num_classes:int, group_num_classes:int, hidden_size, num_layers):
        super(EndModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            *list(models.resnet34(weights=models.ResNet34_Weights.DEFAULT).children())[:-1]
        )
        
        self.norm1 = nn.LayerNorm(512)
        
        self.gru1 = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size*2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, person_num_classes)
        )
        
        self.norm2 = nn.LayerNorm(512)
        
        self.gru2 = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size*2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, group_num_classes)
        )
        
        self.pool = nn.AdaptiveMaxPool2d((1,256))
        
    def forward(self, x):
        # x => (batch, 12, 9, 3, 224, 224)
        #      batch, players, seq, channle, height, width
        B, PL, SEQ, C, H, W = x.shape
        x = x.view(B*PL*SEQ, C ,H ,W)
        x1 = self.feature_extractor(x) #(B*PL*SEQ, 512, 1, 1)
        
        x1 = x1.view(B*PL, SEQ, 512, -1)
        x1 = self.norm1(x1)
        x2, _ = self.gru1(x1) #(B*PL, SEQ, 512*2) 512*2 because it is bidirectional
        x2 = x2[:, -1, :] #Take the last hidden
        
        y1 = self.fc1(x2) #(B*PL, person_classes)
        
        
        x = torch.cat((x1, x2), dim=2) # (B*PL, SEQ, 512*3)
        x = x.contiguous()
        
        
        x = x.view(B*SEQ, PL, -1) #(B*SEQ, PL, 512*3)
        team1 = x[:, :6, :] #(B*SEQ, 6, 512*3)
        team2 = x[:, 6:, :] #(B*SEQ, 6, 512*3)
        
        team1 = self.pool(team1) #(B*SEQ, 1, 256)
        team2 = self.pool(team2) #(B*SEQ, 1, 256)
        x = torch.cat([team1, team2], dim=1)  #(B*SEQ, 2, 256)
        
        x = x.view(B, SEQ, -1) #(B, SEQ, 512)
        x = self.norm2(x) #(B, SEQ, 512)
        x, _ = self.gru2(x) #(B, SEQ, 512*2)
        x = x[:, -1, :] #(B, 512*2) Take the last state
        
        y2 = self.fc2(x) #(B, group_classes)
        
        return {'person': y1, 'group': y2}

def collate_fn(batch):
    clips, person_labels, group_labels  = zip(*batch)  
    
    max_bboxes = 12  
    padded_clips = []
    padded_person_labels = []

    for clip, label in zip(clips, person_labels) :
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_padding = torch.zeros((max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3), clip.size(4)))
            label_padding = torch.zeros((max_bboxes - num_bboxes, label.size(1), label.size(2)))
            
            clip = torch.cat((clip, clip_padding), dim=0)
            label = torch.cat((label, label_padding), dim=0)
            
        padded_clips.append(clip)
        padded_person_labels.append(label)
    
    padded_clips = torch.stack(padded_clips)
    padded_person_labels = torch.stack(padded_person_labels)
    group_labels = torch.stack(group_labels)
    
    group_labels = group_labels[:,-1, :] # # utils the label of last frame
    padded_person_labels = padded_person_labels[:, :, -1, :]  # utils the label of last frame for each player
    b, bb, num_class = padded_person_labels.shape # batch, bbox, num_clases
    padded_person_labels = padded_person_labels.view(b*bb, num_class)

    return padded_clips, padded_person_labels, group_labels


def get_sampler_weights(dataset):
    labels = []
    for idx in range(len(dataset)):
        _, _, group_label = dataset[idx]
        labels.append(group_label[-1].argmax().item()) # take one label of the 9 frame 
    
    class_counts = torch.bincount(torch.tensor(labels))

    num_samples = len(labels)
    class_weights = (1.0 / class_counts.float()) 
    class_weights = class_weights / class_weights.sum()
    
    samples_weights = [class_weights[label] for label in labels]
    samples_weights = torch.DoubleTensor(samples_weights)
    
    return samples_weights, class_weights