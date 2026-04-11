import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from src.config.settings import CLASS_TO_LABEL, ACTIVE_CLASSES


def final_evaluation(
    model:  nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> tuple[float, list, list]:
    """Print classification report and return (accuracy, preds, labels)."""
    model = model.to(device)
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch   = X_batch.to(device)
            outputs   = model(X_batch)
            predicted = torch.argmax(outputs, dim=1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(y_batch.numpy())

    acc        = float(accuracy_score(labels, preds))
    label_names = [CLASS_TO_LABEL[c] for c in ACTIVE_CLASSES]

    print(f"\nFinal Accuracy: {acc:.4f}")
    print(classification_report(labels, preds, target_names=label_names))
    return acc, preds, labels


def plot_history(history: dict, title_prefix: str = "") -> None:
    """Plot loss curves and validation accuracy from a training history dict."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title(f"{title_prefix} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["val_acc"], label="Val Accuracy", color="green")
    axes[1].set_title(f"{title_prefix} Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
