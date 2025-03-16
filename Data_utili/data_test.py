import sys
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os

# Add the path to the data utility module if necessary
sys.path.insert(0, r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\Data_utili")

# Import the dataset class
from DataLoader import VolleyDatasets  # Replace with the actual script name

# Set dataset parameters
dataset_root = r"D:/project/Python/DL(Mostafa saad)/Project/VolleyBall/Data"
split_type = 'train'  # Can be 'train', 'val', or 'test'
mode = 'player_action'  # Can be 'player_action', 'group_activity', or 'player_features_extraction'

# Initialize dataset
print("Initializing dataset...")
dataset = VolleyDatasets(dataset_root=dataset_root, split_type=split_type, mode=mode, use_all_frames=False)

# Check dataset length
print(f"Dataset size: {len(dataset)}")

# Create DataLoader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Fetch a batch
print("Fetching a batch from DataLoader...")
sample_batch = next(iter(dataloader))

if mode in ['player_action', 'group_activity']:
    images, labels = sample_batch
    print(f"Batch image shape: {images.shape}")  # Should be (batch_size, 3, 224, 224)
    print(f"Batch labels: {labels}")

    # Display some images
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(make_grid(images[:batch_size], nrow=4).permute(1, 2, 0))
    ax.axis('off')
    plt.show()

elif mode == 'player_features_extraction':
    frames_data = sample_batch['frames_data']
    label = sample_batch['label']
    meta = sample_batch['meta']

    print(f"Frames batch size: {len(frames_data)}")
    print(f"Sample metadata: {meta}")

print("DataLoader test completed successfully!")
