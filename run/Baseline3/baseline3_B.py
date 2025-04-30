import sys
import os
# adding Folder_2/subfolder to the system path
sys.path.insert(0, r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall")
from Model_classes.Baseline3.baseline3_B import FeatureExtractor
from Data_utili.DataLoader import do_dataLoader
from Handeler.load_save import load_config
import torch
from tqdm import tqdm
import numpy as np
import pickle

config = load_config(r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\config\base3_B_config.yml")
if __name__ == '__main__':
    device = config["device"]
    
    Feature_model = FeatureExtractor(config["beast_model_path"]).to(device=device)
    Feature_model.eval()    
    
    for split in ['train', 'val', 'test']:
        dataloader = do_dataLoader(
            data_path=config["dataset_path"],
            split_type=split,
            batch_size=config["batch_size"],
            mode="player_features_extraction",
            num_workers=config["num_workers"],
            shuffle=False,
            use_all_frames=True,
            pin_memory=True
        )
        split_features = {}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Get Feature From {split}"):
                fram_data = batch["frames_data"]
                label = batch["label"].item()
                meta = batch["meta"]
                
                per_clip_feature=[]
                
                for frame in fram_data:
                    players_croped = frame[0]
                    players_croped = players_croped.to(device)
                    
                    
                    if players_croped.shape[0] == 0:
                        raise ValueError("players_croped is empty")
                    
                    players_features = Feature_model(players_croped)
                    
                    
                    while players_features.shape[0] < 12:
                        players_features = torch.cat((players_features, torch.zeros((1, 2048)).to(device)), dim=0)
                        
                    per_clip_feature.append(players_features.cpu().numpy())
                        
                #print("before the stacking :" ,len(per_clip_feature))
                per_clip_feature = np.stack(per_clip_feature)
                
                key = f"{meta['video_id'][0]}_{meta['clip_id'][0]}"
                split_features[key] = {
                    'features': per_clip_feature,
                    'label': label
                }
        save_path = os.path.join(config['save_dir'], f"{split}_features.pt")
        
        torch.save(split_features, save_path)
            
        print(f"Saved {len(split_features)} {split} clips to {save_path}")


# Visualization
# fig, axes = plt.subplots(3, 4, figsize=(12, 8))
# fig.suptitle(f"Middle Frame {middle_idx} - Padded Players Visualization")

# for i, ax in enumerate(axes.flat):
#     if i < players_croped.shape[0]:
#         img = to_pil(players_croped[i].cpu())  # Convert tensor to image
#         ax.imshow(img)
#         ax.set_title(f"Player {i+1}")
#     ax.axis("off")

# plt.show()

#print(f"Extracted features from middle frame (index {middle_idx})")
#import sys
#import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans

# # Add project path
#sys.path.insert(0, r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall")
# from Model_classes.Baseline3.baseline3_B import FeatureExtractor
# from Data_utili.DataLoader import do_dataLoader
#from Handeler.load_save import load_config

# Load config
#config = load_config(r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\config\base3_B_config.yml")

# def cluster_features(features, n_clusters=8):
#     """Cluster features using KMeans and visualize using t-SNE."""
#     print(f"Clustering features with shape: {features.shape}")  # Debugging print
#     if features.shape[0] < n_clusters:
#         print(f"Warning: Fewer samples ({features.shape[0]}) than clusters ({n_clusters})!")

#     # Apply KMeans clustering
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#     cluster_labels = kmeans.fit_predict(features)

#     # Reduce dimensions with t-SNE
#     tsne = TSNE(n_components=2, perplexity=min(5, features.shape[0] - 1), random_state=42)
#     reduced_features = tsne.fit_transform(features)

#     # Plot t-SNE results
#     plt.figure(figsize=(10, 7))
#     scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap="jet", alpha=0.6)
#     plt.colorbar(scatter, label="Cluster Labels")
#     plt.title("t-SNE Visualization of Clustered Features")
#     plt.show()

#     return cluster_labels

# def extract_features(split, model, device):
#     """Extract features and apply clustering before saving."""
#     dataloader = do_dataLoader(
#         data_path=config["dataset_path"],
#         split_type=split,
#         batch_size=config["batch_size"],
#         mode="player_features_extraction",
#         num_workers=config["num_workers"],
#         shuffle=False,
#         use_all_frames=True,
#         pin_memory=True
#     )

#     features_list, labels_list = [], []
#     split_features = {}

#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc=f"Extracting Features from {split}"):
#             fram_data = batch["frames_data"]
#             label = batch["label"].item()
#             meta = batch["meta"]

#             per_clip_feature = []

#             for frame in fram_data:
#                 players_croped = frame[0].to(device)
#                 players_features = model(players_croped)

#                 while players_features.shape[0] < 12:
#                     players_features = torch.cat((players_features, torch.zeros((1, 2048)).to(device)), dim=0)

#                 per_clip_feature.append(players_features.cpu().numpy())
            
#             per_clip_feature = np.stack(per_clip_feature)  # Shape: (frames, players, 2048)
            
#             # Ensure features are consistently stored as (frames * players, 2048)
#             split_features[f"{meta['video_id'][0]}_{meta['clip_id'][0]}"] = {
#                 'features': per_clip_feature.reshape(-1, 2048),  # Flatten frames & players
#                 'label': label
#             }

#             features_list.append(per_clip_feature.reshape(-1, 2048).flatten())  # Store as flat vector
#             labels_list.append(label)

#     features_np = np.array(features_list)
#     labels_np = np.array(labels_list)

#     print(f"Before saving - Features shape: {features_np.shape}, Labels shape: {labels_np.shape}")

#     # Cluster features before saving
#     print("Clustering Features Before Saving...")
#     cluster_labels = cluster_features(features_np)

#     # Save extracted features
#     save_path = os.path.join(config['save_dir'], f"{split}_features.pt")
#     torch.save(split_features, save_path)
#     print(f"Saved {len(split_features)} {split} clips to {save_path}")

#     return features_np, labels_np

# def load_saved_features(split):
#     """Load saved features after extraction."""
#     load_path = os.path.join(config['save_dir'], f"{split}_features.pt")
#     loaded_data = torch.load(load_path, weights_only=False)

#     features_list, labels_list = [], []

#     for key, value in loaded_data.items():
#         print(f"Key: {key}, Original feature shape: {value['features'].shape}")  # Debugging print
#         features_list.append(value["features"].flatten())  # Flatten to match original shape
#         labels_list.append(value["label"])

#     features = np.array(features_list)
#     labels = np.array(labels_list)

#     print(f"After loading - Features shape: {features.shape}, Labels shape: {labels.shape}")
#     return features, labels

# if __name__ == '__main__':
#     device = config["device"]
#     Feature_model = FeatureExtractor(config["beast_model_path"]).to(device=device)
#     Feature_model.eval()

#     # Extract, cluster, and save features
#     features, labels = extract_features('train', Feature_model, device)

#     # Load saved features
#     print("Loading saved features...")
#     features, labels = load_saved_features('train')

#     # Cluster loaded features
#     print("Clustering Features After Loading...")
#     cluster_labels = cluster_features(features)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torchvision import models
# from torch.optim.lr_scheduler import OneCycleLR
# from sklearn.utils.class_weight import compute_class_weight
# import numpy as np

# class ImprovedCNNModel(nn.Module):
#     def __init__(self, num_classes):
#         super(ImprovedCNNModel, self).__init__()
#         self.backbone = models.resnet50(pretrained=True)
#         self.backbone.fc = nn.Identity()  # Remove FC layer to get features
        
#         self.pool = nn.AdaptiveAvgPool2d((1, 2048))  # Changed from MaxPool to AvgPool
        
#         self.fc_layers = nn.Sequential(
#             nn.Linear(2048, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Dropout(0.4),  # Reduced dropout slightly
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_classes)
#         )
    
#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.pool(x).squeeze()
#         x = self.fc_layers(x)
#         return x


# def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-3, device='cuda'):
#     optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # Added weight decay
#     criterion = nn.CrossEntropyLoss()
#     scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=num_epochs)
    
#     model.to(device)
#     best_val_acc = 0.0
    
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss, correct, total = 0.0, 0, 0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
            
#             running_loss += loss.item() * inputs.size(0)
#             _, preds = torch.max(outputs, 1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
        
#         train_acc = correct / total
#         val_acc = evaluate_model(model, val_loader, device)
        
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/total:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), "best_model.pth")


# def evaluate_model(model, loader, device='cuda'):
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#     return correct / total

# # Example usage:
# model = ImprovedCNNModel(num_classes=10)
# trian_path = os.path.join(config['dataset_path'], 'train_features.pt')
# train_loader = do_dataLoader(trian_path, 'train', 'player_features',batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True, pin_memory=config['pin_memory'])

# val_path = os.path.join(config['dataset_path'], 'val_features.pt')
# val_loader = do_dataLoader(val_path, 'val', 'player_features', batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False, pin_memory=config['pin_memory'])

# train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-3)



