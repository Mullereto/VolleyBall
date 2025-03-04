import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


class Trainer:
    def __init__(self, model, config ,train_loader, val_loader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(self.config["device"])
        self.model = model.to(self.device)
        self.epoch = self.config["epoch"]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, mode="min", factor=0.1, patience=2, cooldown=1, min_lr=1e-6, verbose=True
        )
        self.scaler = GradScaler(enabled=False)

    def train(self):
        best_val_f1 = 0

        for epoch in range(self.epoch):
            self.model.train()
            running_loss = 0.0
            all_labels, all_predicted = [], []

            for img, label in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epoch} [Training]"):
                img, label = img.to(self.device), label.to(self.device)

                self.optimizer.zero_grad()
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
            train_acc = accuracy_score(all_labels, all_predicted)  # Accuracy Calculation
            train_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Train F1 Score: {train_f1:.4f} | Train Accuracy: {train_acc:.4f}")

            # Validation
            val_loss, val_f1, val_acc = self.validate()

            # Update LR based on validation
            self.scheduler.step(val_loss)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), f"{self.config['save_dir']}/best_model.pth")
                print(f"Epoch {epoch + 1}: Best model saved with Validation F1 Score: {val_f1:.4f}")

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_labels, all_predicted = [], []

        with torch.no_grad():
            for img, label in self.val_loader:
                img, label = img.to(self.device), label.to(self.device)

                output = self.model(img)
                loss = self.criterion(output, label)

                running_loss += loss.item()
                _, predicted = torch.max(output, 1)

                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        val_f1 = f1_score(all_labels, all_predicted, average="weighted")
        val_acc = accuracy_score(all_labels, all_predicted)  # Accuracy Calculation
        avg_val_loss = running_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f} | Validation F1 Score: {val_f1:.4f} | Validation Accuracy: {val_acc:.4f}")
        return avg_val_loss, val_f1, val_acc
