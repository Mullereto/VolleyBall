import sys
# adding Folder_2/subfolder to the system path
sys.path.insert(0, r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall")
from Model_classes.Baseline3.baseline3_A import Baseline3
from Data_utili.DataLoader import do_dataLoader
from trainerClass.trainer import Trainer
from trainerClass.Evaluator import Evaluator
from Handeler.load_save import load_config

config = load_config(r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\config\base3_A_config.yml")

if __name__ == '__main__':
    train_loader = do_dataLoader(config['dataset_path'], 'train', 'player_action',batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)
    val_loader = do_dataLoader(config['dataset_path'], 'val', 'player_action', batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=config["shuffle"])
    test_loader = do_dataLoader(config['dataset_path'], 'test', 'player_action', batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=config["shuffle"])
    model = Baseline3()
    
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.train()
    
    evalutor = Evaluator(model, config, test_loader)
    evalutor.evaluate()
    