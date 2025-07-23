import sys
import torch
import os
# adding Folder_2/subfolder to the system path
sys.path.insert(0, r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall")
from Model_classes.Baseline6 import Baseline6, featuregeter
from Data_utili.DataLoader import do_dataLoader
from trainerClass.trainer import Trainer
from trainerClass.Evaluator import Evaluator
from Handeler.load_save import load_config



config = load_config(r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\config\base6_config.yml")


if __name__ == '__main__':

    train_loader = do_dataLoader(config["dataset_path"], 'train', 'player_features_extraction',batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False, use_all_frames=True, pin_memory=config['pin_memory'] ,crop_seq=True)
    
    val_loader = do_dataLoader(config["dataset_path"], 'val', 'player_features_extraction', batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False, use_all_frames=True, pin_memory=config['pin_memory'] ,crop_seq=True)

    test_loader = do_dataLoader(config["dataset_path"], 'test', 'player_features_extraction', batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False, use_all_frames=True, pin_memory=config['pin_memory'] ,crop_seq=True)
    
    featuregeter_model = featuregeter()
    check_point = torch.load(config['state_dict'], map_location=config['device'])
    featuregeter_model.load_state_dict(check_point, strict=False)
    
    model = Baseline6(featuregeter=featuregeter_model)
    
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.train()
    
    evalutor = Evaluator(model, config, test_loader)
    evalutor.evaluate()
    