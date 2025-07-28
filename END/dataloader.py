import sys
from sklearn.model_selection import train_test_split
# adding Folder_2/subfolder to the system path
#sys.path.insert(0, r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall")
import pickle
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
from torch.utils.data import Dataset
import cv2
from PIL import Image
import torch
import numpy as np
from typing import List, Tuple
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
from boxinfo import BoxInfo

class EndDataSet(Dataset):
    """Dataset class for VolleyBall videos

    Args:
        Dataset (Dataset): inherent from torch (Dataset)
    
    Parameters:
        dataset_root (str): the root for the dataset
        split_type (str): the split type neither (train, val, test)
        
    Attribute:
        splits (dict): contain the id of the videos splited to (train, val, test)
        lables (dict): contain the action in the videos
        annot (List): contain the annnotations of all the videos
        samples (List[dict]): contain the samples and its meta data
        transform (transformer): the transformer that will be applied on the data 
    """
    def __init__(self, dataset_root:str, split_type:str):
        self.dataset_root = dataset_root
        self.split_type = split_type
        self.splits = {
            'train' : [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
            'val' : [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
            'test' : [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47],
        }

        self.lables =[{                                         #first element in the list is for player action
                'blocking': 0,                                  #second element in the list is for group activity
                'digging': 1, 
                'falling': 2, 
                'jumping': 3,
                'moving': 4, 
                'setting': 5, 
                'spiking': 6, 
                'standing': 7, 
                'waiting': 8
            },
            {
                'l-pass': 0,
                'r-pass': 1,
                'l-spike': 2,
                'r_spike': 3,
                'l_set': 4,
                'r_set': 5,
                'l_winpoint': 6,
                'r_winpoint': 7
            }
            ]
        self.annot = self.__load_annotations()
        self.samples, self.class_count = self._generate_samples()
        

        self.transform = self.do_transform(split=self.split_type)

    

    def __load_annotations(self):
        annot_path = os.path.join(self.dataset_root, 'annot_all_3frames.pkl')
        if not os.path.exists(annot_path):
            raise FileNotFoundError(f"Annotation file not found: {annot_path}")
        with open(f'{self.dataset_root}/annot_all_3frames.pkl', 'rb') as file:
            v = pickle.load(file)
            return v
    
    def _generate_samples(self):
        sambles = []
        class_count = defaultdict(int)
        
        for clip_id in self.splits[self.split_type]:
            clip_dirs = self.annot[str(clip_id)]
            
            for clip_dir in clip_dirs.keys():
                category = clip_dirs[str(clip_dir)]['category']
                dir_frame = list(clip_dirs[str(clip_dir)]['frame_boxes_dct'].items())
                
                frames_data = []
                
                for frame_id, player_boxs in dir_frame:
                    frame_path = f"{self.dataset_root}/videos/{clip_id}/{clip_dir}/{frame_id}.jpg"
                    
                    frames_boxs = []
                    
                    for box in player_boxs:
                        frames_boxs.append(box)

                    frames_data.append((frame_path, frames_boxs))
                    
                sambles.append({
                    "frames_data":frames_data,
                    'category':category
                })
                class_count[category] += 1

        return sambles, class_count
    
        
    def _calculate_box_center(self, box: BoxInfo):
    
        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2

        return  x_center
    
    def extract_person_crops(self, frame:np.ndarray, boxes:list[BoxInfo]):
        crops: List = []
        order: List = []
        person_frame_labels : List = []
        
        for box in boxes:
            x_min, y_min, x_max, y_max = box.box
            x_center = self._calculate_box_center(box.box)
            
            person_crops = frame[y_min:y_max, x_min:x_max]
            
            if self.transform:
                transformed = self.transform(image=person_crops)
                person_crops = transformed['image']
                
            person_label = torch.zeros(len(self.lables[0]))
            person_label[self.lables[0][box.category]] = 1  
            crops.append(person_crops)
            order.append(x_center)  # or x_center if you want to order by center
            person_frame_labels.append(person_label)        
            
        return crops, order, person_frame_labels
    
    def do_transform(self, split):
        if split == 'train':
            transformer = A.Compose([
                A.Resize(224, 224),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.ColorJitter(brightness=0.2),
                    A.RandomBrightnessContrast(),
                    A.GaussNoise(),
                    A.MotionBlur(blur_limit=5), 
                    A.MedianBlur(blur_limit=5)  
                ], p=0.55),
                A.OneOf([
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                    A.RandomRotate90()
                ], p=0.01),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            transformer = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])  
        return transformer
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        group_label  = torch.zeros(len(self.lables[1]))
        group_label [self.lables[1][sample['category']]] = 1
        
        clip = []
        person_labels = []
        group_labels = []
        
        for frame_path, boxes in sample['frames_data']:
            frame = cv2.imread(frame_path)
            
            crops, order, person_frame_labels = self.extract_person_crops(frame, boxes)
            sorted_players = sorted(zip(order, crops, person_frame_labels), key=lambda x: x[0])
            
            sorted_crops = [crop for _, crop, _ in sorted_players]
            sorted_labels = [label for _, _, label in sorted_players]
            
            
            crops = torch.stack(sorted_crops)
            sorted_labels = torch.stack(sorted_labels)
            person_labels.append(sorted_labels)
            
            
            clip.append(crops)
            group_labels.append(group_label)
        # (9, 12, 3, 224, 224) ==> (12, 9, 3, 224, 224)
        clip = torch.stack(clip).permute(1, 0, 2, 3, 4)
        
        person_labels = torch.stack(person_labels)#(9, 12, 9)(FRAMES, PLAYERS, CLASSES)
        person_labels = person_labels.permute(1, 0, 2)
        group_labels = torch.stack(group_labels) #(9, 8)(FRAMES, CLASSES)

        return clip, person_labels, group_labels
          
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

    class_weights = (1.0 / class_counts.float()) 
    class_weights = class_weights / class_weights.sum()
    
    return class_weights



    
if __name__ == '__main__':
    #d = FeaturesData(r'D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\results\baseline3\phase2\train_features.pt')
    #c = FeaturesData(r'D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\results\baseline3\phase2\val_features.pkl')
    data_path = r'D:/project/Python/DL(Mostafa saad)/Project/VolleyBall/Data'
    d = EndDataSet(data_path, 'train', 'player_action')
    

    #train_loader = do_dataLoader(data_path=data_path,split_type= 'train',mode= 'player_features_extraction',batch_size=16, num_workers=4, shuffle=True, pin_memory=True, crop_seq=True, use_all_frames=True)
    dataloader_test = DataLoader(dataset=d, batch_size=1, num_workers=4, shuffle=False)
    
    for clip in dataloader_test:
        print(clip.shape)
        break
