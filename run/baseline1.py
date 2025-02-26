import sys
# adding Folder_2/subfolder to the system path
sys.path.insert(0, r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall")
from Model_classes.baseline1 import Baseline1
from Data_utili.DataLoader import do_dataLoader
from trainerClass.trainer import Trainer
from trainerClass.Evaluator import Evaluator
from Handeler.load_save import load_config

config = load_config(r"D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\config\base1_config.yml")

if __name__ == '__main__':
    train_loader = do_dataLoader(config['dataset_path'], 'train', 'group_activity',batch_size=config['batch_size'], num_workers=config['num_workers'])
    val_loader = do_dataLoader(config['dataset_path'], 'val', 'group_activity', batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)
    test_loader = do_dataLoader(config['dataset_path'], 'test', 'group_activity', batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)
    model = Baseline1()
    
    trainer = Trainer(model, config, train_loader, val_loader)
    trainer.train()
    
    evalutor = Evaluator(model, config, test_loader)
    evalutor.evaluate()
    