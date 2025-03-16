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
    
    
    
    for split in ['train','val','test']:
        dataloader = do_dataLoader(
            data_path=config["dataset_path"],
            split_type=split,
            batch_size=config["batch_size"],
            mode="player_features_extraction",
            num_workers=config["num_workers"],
            shuffle=False,
            use_all_frames=True
        )
        split_features = {}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Get Feature From {split}"):
                fram_data = batch["frames_data"]
                label = batch["label"]
                meta = batch["meta"]
                
                per_clip_feature=[]
                
                for frame in fram_data:
                    players_croped = frame[0]
                    players_croped = players_croped.to(device)
                    
                    
                    players_features = Feature_model(players_croped)
                    while players_features.shape[0] < 12:
                        players_features = torch.cat((players_features, torch.zeros((1, 2048)).to(device)), dim=0)
                        
                    per_clip_feature.append(players_features.cpu().numpy())
                    
                
                per_clip_feature = np.stack(per_clip_feature)
                
                key = f"{meta["video_id"][0]}"
                
                split_features[key] = {
                    'features': per_clip_feature,
                    'label': label
                }
    
        save_path = os.path.join(config['save_dir'], f"{split}_features.pkl")
        
        with open(save_path, 'wb') as f:
            pickle.dump(split_features, f)
            
        print(f"Saved {len(split_features)} {split} clips to {save_path}")
        

# import sys
# import os
# import torch
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans

# # Add project path
# sys.path.insert(0, r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall")
# from Model_classes.Baseline3.baseline3_B import FeatureExtractor
# from Data_utili.DataLoader import do_dataLoader
# from Handeler.load_save import load_config

# # Load config
# config = load_config(r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\config\base3_B_config.yml")

# def extract_features(split, model, device):
#     dataloader = do_dataLoader(
#         data_path=config["dataset_path"],
#         split_type=split,
#         batch_size=config["batch_size"],
#         mode="player_features_extraction",
#         num_workers=config["num_workers"],
#         shuffle=False,
#         use_all_frames=True
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
#                 features_list.append(players_features.cpu().numpy().flatten())
#                 labels_list.append(label)
            
#             per_clip_feature = np.stack(per_clip_feature)
#             key = f"{meta['video_id'][0]}"
            
#             split_features[key] = {'features': per_clip_feature, 'label': label}
    
#     # Save extracted features
#     # save_path = os.path.join(config['save_dir'], f"{split}_features.pkl")
#     # with open(save_path, 'wb') as f:
#     #     pickle.dump(split_features, f)
#     # print(f"Saved {len(split_features)} {split} clips to {save_path}")
    
#     return np.array(features_list), np.array(labels_list)


# def visualize_features(features, labels):
#     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#     reduced_features = tsne.fit_transform(features)
    
#     plt.figure(figsize=(10, 7))
#     scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap="jet", alpha=0.6)
#     plt.colorbar(scatter, label="Class Labels")
#     plt.title("t-SNE Visualization of Extracted Features")
#     plt.show()


# def cluster_features(features, n_clusters=9):
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     cluster_labels = kmeans.fit_predict(features)
    
#     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#     reduced_features = tsne.fit_transform(features)
    
#     plt.figure(figsize=(10, 7))
#     scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap="jet", alpha=0.6)
#     plt.colorbar(scatter, label="Cluster Labels")
#     plt.title("t-SNE Visualization of Clustered Features")
#     plt.show()
    
#     return cluster_labels


# if __name__ == '__main__':
#     device = config["device"]
#     Feature_model = FeatureExtractor(config["beast_model_path"]).to(device=device)
#     Feature_model.eval()
    
#     features, labels = extract_features('train', Feature_model, device)
    
#     print("Visualizing Extracted Features...")
#     visualize_features(features, labels)
    
#     print("Clustering Extracted Features...")
#     cluster_labels = cluster_features(features)
