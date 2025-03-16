import sys
import os
# adding Folder_2/subfolder to the system path
sys.path.insert(0, r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall")
from Model_classes.Baseline3.Baseline3_C import NNClassifier
from Data_utili.DataLoader import do_dataLoader
from trainerClass.trainer import Trainer
from trainerClass.Evaluator import Evaluator
from Handeler.load_save import load_config

config = load_config(r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\config\base3_C_config.yml")

if __name__ == '__main__':
    trian_path = os.path.join(config['dataset_path'], 'train_features.pkl')
    train_loader = do_dataLoader(trian_path, 'train', 'player_features',batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True)
    
    val_path = os.path.join(config['dataset_path'], 'val_features.pkl')
    val_loader = do_dataLoader(val_path, 'val', 'player_features', batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)
    
    test_path = os.path.join(config['dataset_path'], 'test_features.pkl')
    test_loader = do_dataLoader(test_path, 'test', 'player_features', batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)
    
    
    model = NNClassifier()
    
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.train()
    
    evalutor = Evaluator(model, config, test_loader)
    evalutor.evaluate()
    