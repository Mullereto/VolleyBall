import sys
from sklearn.model_selection import train_test_split
# adding Folder_2/subfolder to the system path
sys.path.insert(0, r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\Data_utili")
import pickle
import os
from torchvision import transforms
from collections import defaultdict
from torch.utils.data import Dataset
import cv2
from PIL import Image
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt

class VolleyDatasets(Dataset):
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
    def __init__(self, dataset_root:str, split_type:str, mode:str, use_all_frames = False):
        self.dataset_root = dataset_root
        self.split_type = split_type
        self.splits = {
            'train' : [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
            'val' : [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
            'test' : [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47],
        }
        self.mode = mode
        self.use_all_frames = use_all_frames
        self.annot = self.__load_annotations()
        if self.mode == 'player_action':
            self.lables = {
                'waiting' : 0,
                'setting' : 1,
                'digging': 2,  
                'falling': 3,  
                'spiking': 4,  
                'blocking': 5,  
                'jumping': 6,  
                'moving': 7,  
                'standing': 8,  
            }
        else:
            self.lables = {
                'l-pass': 0,
                'r-pass': 1,
                'l-spike': 2,
                'r_spike': 3,
                'l_set': 4,
                'r_set': 5,
                'l_winpoint': 6,
                'r_winpoint': 7
            }    
        self.samples, self.class_count = self._generate_samples()
        self.class_weights = self._compute_class_weights()
        if self.mode == 'player_action':
            self.sample_weights = [
                self.class_weights[s['action']] if 'action' in s else 1.0 for s in self.samples
            ]
        elif self.mode == 'group_activity':
            self.sample_weights = [
                self.class_weights[s['activity']] if 'activity' in s else 1.0 for s in self.samples
            ]
        else:
            self.sample_weights = None
        self.annot = self.__load_annotations()
        
        if self.mode == 'player_action' or self.mode == 'group_activity':
            self.transform = self.do_transform(split=self.split_type)

        else:
            self.transform = self.do_tranform_featuers()
    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.mode in ['player_action', 'group_activity']:
            try:
                frame_path = os.path.join(self.dataset_root, 'videos',sample['vid_id'], sample['clip_id'], f"{sample['frame_id']}.jpg")
                frame = cv2.imread(frame_path) 
                if frame is None:
                    raise ValueError(f"Could not load image: {frame_path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.mode == 'player_action':    
                    x1,y1,x2,y2 = map(int, sample['box'])
                    h, w, _ = frame.shape
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
                    image = frame[y1:y2, x1:x2]
                    if image.size == 0:
                        raise ValueError("Empty crop")
                elif self.mode == 'group_activity':
                    image = frame
            except Exception as e:
                print(f"Error loading sample {idx}: {e}")
                image = np.zeros((224, 224, 3), dtype=np.uint8)
        
            image = Image.fromarray(image)
            # plt.imshow(image)
            # plt.axis('off')  # Hide axes
            # plt.title(f"Label: {self.lables[sample['player_action']]}", fontsize=12, color='red', fontweight='bold')
            # plt.show()    
            image = self.transform(img=image)

            return image, self.lables[sample['player_action']] if self.mode == 'player_action' else self.lables[sample['group_activity']]
        
        elif self.mode == 'player_features_extraction':
            video_path = os.path.join(self.dataset_root, 'videos',sample['vid_id'], sample['clip_id'])
            
            frame_detel = []
            
            for frame_id in sample['frame_id']:
                #print("////the frame_id////")
                #print(frame_id)
                frame_boxes = self.annot[sample['vid_id']][sample['clip_id']]['frame_boxes_dct'][frame_id]
                
                frame_boxes.sort(key=lambda box:box.box[0])
                
                player_corped = []
                player_corped_id = []
                for box in frame_boxes:
                    try:
                        frame_path = os.path.join(video_path, f'{frame_id}.jpg')
                        frame = cv2.imread(frame_path)
                        if frame is None:
                            continue
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        x1, y1, x2, y2 = map(int, box.box)
                        player_crop = frame[y1:y2, x1:x2]
                        player_crop = Image.fromarray(player_crop)

                        player_crop = self.transform(player_crop)
                        
                        player_corped.append(player_crop)
                        player_corped_id.append(box.player_ID)
                    except Exception as e:
                        print(f"Error processing crop: {e}")
                        continue                        
                # max_players = 12  # Set this to the maximum number of players per frame
                # while len(player_corped) < max_players:
                #     player_corped.append(torch.zeros((3, 224, 224)))    
                
                frame_detel.append(torch.stack(player_corped))
                #print(f" the len is {len(frame_detel)}")
            return {
                'frames_data': frame_detel,
                'label': self.lables[sample['activity']],
                'meta': {
                    'video_id': sample['vid_id'],
                    'clip_id': sample['clip_id'],
                    'frame_ids': sample['frame_id']
                }
            }        
    def _compute_class_weights(self):
        """
           Compute class weights for balancing the dataset.

           Returns:
           - dict: A dictionary where keys are class labels and values are weights.
        """
        total_samples = sum(self.class_count.values())
        if total_samples == 0:
            return {action: 1.0 for action in self.lables.keys()}
        return {
            action: total_samples / (len(self.lables) * count)
            for action, count in self.class_count.items()
            if count > 0
        }                
        
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
        
        for vid_id in map(str, self.splits[self.split_type]):
            if vid_id not in self.annot:
                print(f"Warning: Video {vid_id} not found in annotations")
                continue
            for clip_id, clip_data in self.annot[vid_id].items():
                frame_ids = sorted(clip_data['frame_boxes_dct'].keys())
                selected_frame_ids = frame_ids if self.use_all_frames == True else  [frame_ids[len(frame_ids) // 2]]
                
                if self.mode in ['player_action', 'group_activity']:
                    for fram_id in selected_frame_ids:
                        frame_path = os.path.join(self.dataset_root, 'videos', vid_id, clip_id , f"{fram_id}.jpg")
                        if not os.path.exists(frame_path):
                            print(f"Warrning this frame dose not exist{frame_path}")
                            continue
                        if self.mode == 'group_activity':
                            sambles.append({
                                'vid_id' : vid_id,
                                'clip_id' : clip_id,
                                'frame_id' : fram_id,
                                'group_activity' : clip_data['category'], 
                            })
                            class_count[clip_data['category']]+=1
                            
                        elif self.mode == 'player_action':
                            boxs = clip_data['frame_boxes_dct'][fram_id]
                            for box in boxs:
                                if box.category not in self.lables:
                                    continue
                                sambles.append({
                                    'vid_id' : vid_id,
                                    'clip_id' : clip_id,
                                    'frame_id' : fram_id,
                                    'box' : box.box,
                                    'player_action' : box.category, 
                                })
                                class_count[box.category]+=1
                elif self.mode == 'player_features_extraction':
                    sambles.append({
                                'vid_id' : vid_id,
                                'clip_id' : clip_id,
                                'frame_id' : selected_frame_ids,
                                'activity' : clip_data['category'], 
                            })

        return sambles, class_count
               
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
    
    def do_tranform_featuers(self):
        transformer = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])       
        return transformer


class FeaturesData(Dataset):
    def __init__(self, features_path):
        
        self.data = self.load_features(features=features_path)
        
        features = []
        labels = []
        
        for vid_id, items in self.data.items():
            features.append(items['features'])
            labels.append(items['label'])
        
        self.featrues = features
        self.label = labels
                
    def load_features(self, features):
        with open(features, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.featrues[idx])
        label = torch.LongTensor([self.label[idx]]).squeeze()
        
        #print(f"Feature shape: {features.shape}, Label shape: {label.shape}")
        
        return features, label
            
        


def do_dataLoader(data_path:str, split_type:str, mode:str, batch_size:int, num_workers:int, shuffle = True, use_all_frames=False):
    if mode in ['player_action', 'group_activity', 'player_features_extraction']:
        dataset = VolleyDatasets(dataset_root=data_path, split_type=split_type, mode=mode, use_all_frames=use_all_frames)
        if split_type == 'train' and mode != "player_features_extraction":
            sampler = WeightedRandomSampler(weights=dataset.sample_weights, num_samples=len(dataset.samples), replacement=True)

            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, sampler=sampler)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    elif mode == 'player_features':
        dataset = FeaturesData(features_path=data_path)
        
        dataloader = DataLoader(dataset,batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return dataloader


    
if __name__ == '__main__':
    d = FeaturesData(r'D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\results\baseline3\phase2\train_features.pkl')
    c = FeaturesData(r'D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\results\baseline3\phase2\val_features.pkl')
    dataloader = DataLoader(dataset=d, batch_size=32, num_workers=4, shuffle=True)
    dataloader_test = DataLoader(dataset=c, batch_size=32, num_workers=4, shuffle=False)

        # Convert dataset to NumPy arrays for train-test split
    features_list, labels_list = [], []
    for feature, label in dataloader:
        features_list.append(feature.numpy())  # Convert tensor to numpy
        labels_list.append(label.numpy())
 