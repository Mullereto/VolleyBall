import sys

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
from torch.utils.data import DataLoader

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
        
        self.annot = self.__load_annotations()
        self.samples, self.class_count = self._generate_samples()
        if self.mode == 'player_action' or self.mode == 'group_activity':
            self.transform = self.do_transform()

        else:
            self.transform = self.do_tranform_featuers()
    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.mode in ['player_action', 'group_activity']:
            frame_path = os.path.join(self.dataset_root, 'videos',sample['vid_id'], sample['clip_id'], f"{sample['frame_id']}.jpg")
            frame = cv2.imread(frame_path) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.mode == 'player_action':
                x1,y1,x2,y2 = sample['box']
                
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                image = frame[y1:y2, x1:x2]

            elif self.mode == 'group_activity':
                image = frame
            
            image = Image.fromarray(frame)
            
            image = self.transform(img=image)
            return image, self.lables[sample['player_action']] if self.mode == 'player_action' else self.lables[sample['group_activity']]
        
        elif self.mode == 'player_features_extraction':
            video_path = os.path.join(self.dataset_root, 'videos',sample['vid_id'], sample['clip_id'], sample['frame_id'])
            
            frame_detel = []
            
            for frame in sample['frame_id']:
                frame_boxes = self.annot[sample['vid_id']][sample['clip_id']]['frame_boxes_dct'][frame]
                
                frame_boxes.sort(key=lambda box:box.box[0])
                
                player_corped = []
                player_corped_id = []
                
                for box in frame_boxes:
                    frame_path = os.path.join(video_path, f'{frame}.jpg')
                    frame = cv2.imread(frame_path)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    x1, y1, x2, y2 = map(int, box.box)
                    player_crop = frame[y1:y2, x1:x2]
                    player_crop = Image.fromarray(player_crop)
                    player_crop = self.transform(player_crop)
                    
                    player_corped.append(player_crop)
                    player_corped_id.append(box.player_ID)
                    
                    frame_detel.append(torch.stack(player_corped))
            return {
                'frames_data': frame_detel,
                'label': self.lables[sample['activity']],
                'meta': {
                    'video_id': sample['vid_id'],
                    'clip_id': sample['clip_id'],
                    'frame_ids': sample['frame_id']
                }
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
                selected_frame_id = frame_ids if self.use_all_frames == True else  [frame_ids[len(frame_ids) // 2]]
                if self.mode in ['player_action', 'group_activity']:
                    for fram_id in selected_frame_id:
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
                                'frame_id' : fram_id,
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

def do_dataLoader(data_path:str, split_type:str, mode:str, batch_size:int, num_workers:int, shuffle = True, use_all_frames=False):
    dataset = VolleyDatasets(dataset_root=data_path, split_type=split_type, mode=mode, use_all_frames=use_all_frames)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    
    return dataloader
