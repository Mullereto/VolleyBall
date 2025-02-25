import pickle
import os
from torchvision import transforms

class VolleyDatasets():
    def __init__(self, dataset_root, split_type = 'train'):
        self.dataset_root = dataset_root
        self.split_type = split_type
        self.splits = {
            'train' : [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
            'val' : [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
            'test' : [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47],
        }
        self.load_annot = self.load_annotations()
        self.samples, self.class_count = self.generate_samples()
        self.transform = self.do_transform(split=split_type)
        
        
        
    def __getitem__(self):
        pass
    
    def load_annotations(self):
        annot_path = os.path.join(self.dataset_root, 'annot_all_3frames.pkl')
        if not os.path.exists(annot_path):
            raise 
        with open(f'{self.dataset_root}/annot_all_3frames.pkl', 'rb') as file:
            videos_annot = pickle.load(file)
        
        return videos_annot
    
    
    def generate_samples(self):
        pass
    
    def do_transform(self, split):
        if split == 'train':
            transformer = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomRotation(degrees=20),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transformer = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return transformer