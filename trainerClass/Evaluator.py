import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from Handeler.matricsploter import plot_confusion_matrix, save_classification_report

class Evaluator:
    def __init__(self, model, config, test_loader):
        self.config = config
        self.device = torch.device(self.config["device"])
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self):
        
        self.model.load_state_dict(torch.load(self.config["beast_model_path"], map_location=self.device))
        
        
        self.model.eval()
        running_loss = 0.0
        all_labels, all_predicted = [], []

        with torch.no_grad():
            for img, label in tqdm(self.test_loader, desc="Evaluating on Test Set"):
                img, label = img.to(self.device), label.to(self.device)

                output = self.model(img)
                loss = self.criterion(output, label)

                _, predicted = torch.max(output, 1)
                running_loss += loss.item()
                

                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        test_f1 = f1_score(all_labels, all_predicted, average="weighted")
        test_acc = accuracy_score(all_labels, all_predicted)  
        avg_test_loss = running_loss / len(self.test_loader)

        print(f"Test Loss: {avg_test_loss:.4f} | Test F1 Score: {test_f1:.4f} | Test Accuracy: {test_acc:.4f}")
        save_classification_report(all_labels, all_predicted, self.config["class_names"], 'baseline1')
        plot_confusion_matrix(all_labels, all_predicted, self.config["class_names"], 'baseline1')
