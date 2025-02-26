import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, class_names, baseline_num=None):
    """Plot the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
    
    if baseline_num:
        path = r'D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\results'
        path = os.path.join(path, baseline_num, "Confusion Matrix.jpg")
        plt.savefig(path)
        print(f"Confusion matrix saved to {path}")

def save_classification_report(y_true, y_pred, class_names, baseline_num):
    """Save the classification report to a text file."""
    report = classification_report(y_true, y_pred, target_names=class_names)
    path = r'D:\project\Python\DL(Mostafa saad)\Project\VolleyBall\results'
    path = os.path.join(path, baseline_num, "classification.txt")
    with open(path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {path}")
