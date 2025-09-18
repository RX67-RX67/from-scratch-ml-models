import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support


class Reporter:

    def __init__(self, task:str = 'regression', save_path:str = None):
        self.task = task
        self.save_path = save_path
    
    def plot_loss_curves(self, epoch, train_losses, val_losses):
        plt.figure(figsize=(6,4))
        plt.plot(epoch, train_losses, label="Train Loss")
        plt.plot(epoch, val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curves")
        if self.save_path:
            plt.savefig(f"{self.save_path}/loss_curves.png")
        plt.show()

    def parity_plot(self, y_true, y_pred):
        if self.task != "regression":
            return
        plt.figure(figsize=(6,4))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("Parity Plot")
        if self.save_path:
            plt.savefig(f"{self.save_path}/parity_plot.png")
        plt.show()
    
    def confusion_matrix_plot(self, y_true, y_pred):
        if self.task not in ["binary_classification", "multiclass_classification"]:
            return
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sb.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        if self.save_path:
            plt.savefig(f"{self.save_path}/confusion_matrix.png")
    plt.show()

    def classification_metrics(self, y_true, y_pred):
        if self.task not in ["binary_classification", "multiclass_classification"]:
            return
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")