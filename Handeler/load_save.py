import os
import torch
import yaml

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