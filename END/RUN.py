from dataloader import EndDataSet, collate_fn, get_sampler_weights, DataLoader
from Trainer import Trainer, Validation
from Model import EndModel
from Handler import load_config

if __name__ == '__main__':
    config = load_config(r'END\end_config.yml')
    train_data = EndDataSet(config['dataset_path'], 'train')
    val_data = EndDataSet(config['dataset_path'], 'val')
    test_data = EndDataSet(config['dataset_path'], 'test')
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['num_workers'], pin_memory=config['pin_memory'], collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=config['shuffle'], num_workers=config['num_workers'], pin_memory=config['pin_memory'], collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=config['pin_memory'], collate_fn=collate_fn)

    model = EndModel(person_num_classes=9, group_num_classes=8, hidden_size=512, num_layers=2)
    
    class_weights = get_sampler_weights(train_data)
    trainer = Trainer(model, config, train_loader, val_loader, class_weights)
    trainer.train()
    
    
    validator = Validation(model, config, test_loader)
    validator.eval()
    print("Test completed.")