import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from torchinfo import summary


class Trainer:
    def __init__(self, model, config, train_loader, val_loader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(self.config["device"])
        self.model = model.to(self.device)
        #print("🔍 Model Summary:")
        #summary(model, input_size=(24, 9, 12, 2048), col_names=["input_size", "output_size", "num_params"])
        
        self.epoch = self.config["epoch"]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=1e-3)
        if config["the_scheduler"] == "OneCycleLR":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.config["lr"] * 10,
                steps_per_epoch=len(self.train_loader),
                epochs=self.epoch
            )
        elif config["the_scheduler"] == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer, mode="min", factor=0.1, patience=3, verbose=True
            )

        self.scaler = GradScaler()

    def train(self):
        best_val_f1 = 0

        for epoch in range(self.epoch):
            self.model.train()
            running_loss = 0.0
            all_labels, all_predicted = [], []

            print(f"\nEpoch {epoch + 1}/{self.epoch}")
            print(f"Current LR: {self.optimizer.param_groups[0]['lr']}")

            for img, label in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Training]"):
                # if batch_idx >= overfit_batches:  # Stop after `overfit_batches` batches
                #     break
                
                img, label = img.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()
                print("Feature shape:", img.shape)  # Should be (batch_size, 9, 12, 2048)
                print("Label shape:", label.shape) 
                with autocast():
                    output = self.model(img)
                    loss = self.criterion(output, label)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                all_labels.extend(label.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())

            train_f1 = f1_score(all_labels, all_predicted, average="weighted")
            train_acc = accuracy_score(all_labels, all_predicted)
            train_loss = running_loss / len(self.train_loader)

            print(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f} | Train Acc: {train_acc:.4f}")

            # Validate Model
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
            for img, label in tqdm(self.val_loader, desc="[vallidating]"):
                img, label = img.to(self.device), label.to(self.device)

                output = self.model(img)
                loss = self.criterion(output, label)

                running_loss += loss.item()
                _, predicted = torch.max(output, 1)

                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        val_f1 = f1_score(all_labels, all_predicted, average="weighted")
        val_acc = accuracy_score(all_labels, all_predicted)
        avg_val_loss = running_loss / len(self.val_loader)

        print(f"✅ Validation Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
        return avg_val_loss, val_f1, val_acc
