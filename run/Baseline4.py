import sys
# adding Folder_2/subfolder to the system path
sys.path.insert(0, r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall")
from Model_classes.Baseline4 import Baseline4
from Data_utili.DataLoader import do_dataLoader
from trainerClass.trainer import Trainer
from trainerClass.Evaluator import Evaluator
from Handeler.load_save import load_config

config = load_config(r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\config\base4_config.yml")

if __name__ == '__main__':
    train_loader = do_dataLoader(config['dataset_path'], 'train', 'player_features_extraction',batch_size=config['batch_size'], num_workers=config['num_workers'],shuffle=True, use_all_frames=True, pin_memory=True, sequnce=True)
    val_loader = do_dataLoader(config['dataset_path'], 'val', 'player_features_extraction', batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False, use_all_frames=True, pin_memory=True, sequnce=True)
    test_loader = do_dataLoader(config['dataset_path'], 'test', 'player_features_extraction', batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False, use_all_frames=True, pin_memory=True, sequnce=True)
    model = Baseline4(config['beast_model_path'])
    
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.train()
    
    evalutor = Evaluator(model, config, test_loader)
    evalutor.evaluate()
    