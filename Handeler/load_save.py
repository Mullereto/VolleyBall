import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

def load_config(path:str):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        return config

def save_checkpoint(model, optimizer, epoch, save_dir, filename='checkpoint.pth'):
    """Save model and optimizer state to a checkpoint file."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')
    
    
def load_checkpoint(filepath, model, optimizer=None):
    """Load model and optimizer state from a checkpoint file."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']



def visualize_samples(dataloader, num_samples=5):
    for images, labels in dataloader:
        for i in range(min(num_samples, len(images))):
            image = images[i].permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
            image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))  # Unnormalize
            image = np.clip(image, 0, 1)  # Clip to valid range
            plt.imshow(image)
            plt.title(f"Label: {labels[i]}")
            plt.show()
        break