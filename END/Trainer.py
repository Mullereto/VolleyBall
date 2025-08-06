import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from torchinfo import summary
from Handler import plot_confusion_matrix, save_classification_report, load_config

class Trainer:
    def __init__(self, model, config, train_loader, val_loader, class_weights=None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(self.config["device"])
        self.model = model.to(self.device)
        self.class_weights = class_weights.to(self.device)
        self.epoch = self.config["epoch"]
        self.person_criterion = nn.CrossEntropyLoss()
        self.group_criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["lr"] , weight_decay=1)
        if config["the_scheduler"] == "OneCycleLR":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.config["lr"],
                steps_per_epoch=len(self.train_loader),
                epochs=self.epoch
            )
        elif config["the_scheduler"] == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer, mode="min", factor=0.1, patience=3, verbose=True
            )

        self.scaler = GradScaler()
        # for param in model.feature_extractor.parameters():
        #     param.requires_grad = False
            
        # for param in model.lstm1.parameters():
        #     param.requires_grad = False
            
        # for param in model.lstm2.parameters():
        #     param.requires_grad = False

    def train(self):
        best_val_f1 = 0

        for epoch in range(self.epoch):
            # if epoch == 3:  # e.g. 5
            #     print("🔓 Unfreezing feature extractor and LSTM...")
            #     for param in self.model.feature_extractor.parameters():
            #         param.requires_grad = True
                    
            #     for param in self.model.lstm1.parameters():
            #         param.requires_grad = True
                    
            #     for param in self.model.lstm2.parameters():
            #         param.requires_grad = True
                    
            self.model.train()
            running_loss = 0.0

            all_labels, all_predicted = [], []

            print(f"\nEpoch {epoch + 1}/{self.epoch}")
            print(f"Current LR: {self.optimizer.param_groups[0]['lr']}")

            for img, person_label, group_label in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Training]"):

                img, person_label, group_label = img.to(self.device), person_label.to(self.device), group_label.to(self.device)
                self.optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    output = self.model(img)
                    loss1 = self.person_criterion(output['person'], person_label)
                    loss2 = self.person_criterion(output['group'], group_label)
                    loss = loss2 + (0.4 * loss1)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item()
                predicted = output['group'].argmax(1)
                target_class = group_label.argmax(1)
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(target_class.cpu().numpy())


            train_f1 = f1_score(all_labels, all_predicted, average="weighted")
            train_acc = accuracy_score(all_labels, all_predicted)
            train_loss = running_loss / len(self.train_loader)

            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Train Acc: {train_acc:.4f}")

            # # Validate Model
            val_loss, val_f1, val_acc = self.validate()

            self.scheduler.step(val_loss)  

            print(f"Epoch {epoch+1}: New LR After Scheduler: {self.optimizer.param_groups[0]['lr']}")

            # Save Best Model Based on Validation F1-score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), f"{self.config['save_dir']}/best_model.pth")
                print(f"Best Model Saved! Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_labels, all_predicted = [], []

        with torch.no_grad():
            for img, person_label, group_label in tqdm(self.val_loader, desc="[vallidating]"):
                img, person_label, group_label = img.to(self.device), person_label.to(self.device), group_label.to(self.device)

                output = self.model(img)
                loss1 = self.person_criterion(output['person'], person_label)
                loss2 = self.group_criterion(output['group'], group_label)
                loss = loss2 + (0.30 * loss1)

                running_loss += loss.item()
                
                predicted = output['group'].argmax(1)
                target_class = group_label.argmax(1)
                
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(target_class.cpu().numpy())

        val_f1 = f1_score(all_labels, all_predicted, average="weighted")
        val_acc = accuracy_score(all_labels, all_predicted)
        avg_val_loss = running_loss / len(self.val_loader)

        print(f"✅ Validation Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
        return avg_val_loss, val_f1, val_acc


class Validation:
    def __init__(self, model, config, test_loader):
        self.config = config
        self.device = torch.device(self.config["device"])
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()

    def eval(self):
        self.model.load_state_dict(torch.load(self.config["beast_model_path"], map_location=self.device))
        
        self.model.eval()  
        y_true = []
        y_pred = []
        total_loss = 0.0

        with torch.no_grad():
            for inputs, person_labels, group_labels in tqdm(self.test_loader, desc="[testing]"):
                inputs = inputs.to(self.device)
                person_labels = person_labels.to(self.device)
                group_labels = group_labels.to(self.device)
                
                outputs = self.model(inputs)
                loss_1 = self.criterion(outputs['person'], person_labels)
                loss_2 = self.criterion(outputs['group'], group_labels)
                
                loss = (0.70 * loss_2) + (0.30 * loss_1)
                
                total_loss += loss.item()
                
                _, predicted = outputs['group'].max(1)
                _, target_class = group_labels.max(1)
                
                y_true.extend(target_class.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
        test_f1 = f1_score(y_true, y_pred, average="weighted")
        test_acc = accuracy_score(y_true, y_pred)
        avg_test_loss = total_loss / len(self.test_loader)
        
        print(f"Test Loss: {avg_test_loss:.4f} | Test F1 Score: {test_f1:.4f} | Test Accuracy: {test_acc:.4f}")
        save_classification_report(y_true, y_pred, self.config["class_names"], self.config["save_dir"])
        plot_confusion_matrix(y_true, y_pred, self.config["class_names"], self.config["save_dir"])

