"""
src/training/trainer.py
========================
train() and evaluate() — the training loop and validation routine,
refactored from the notebook into reusable functions.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.data.augmentation import augment


def evaluate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    str = "cpu",
) -> tuple[float, float]:
    """Return (avg_loss, accuracy) on the given DataLoader."""
    model.eval()
    preds, labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs    = model(X_batch)
            loss       = criterion(outputs, y_batch)
            total_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = float(accuracy_score(labels, preds))
    return avg_loss, acc


def train(
    model:      nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    optimizer:    Any,
    criterion:    nn.Module,
    scheduler:    Any | None = None,
    epochs:       int = 50,
    device:       str = "cpu",
) -> dict:
    """
    Training loop with augmentation, validation, and optional LR scheduling.

    Returns
    -------
    history : dict with keys train_loss, val_loss, val_acc (lists per epoch)
    """
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    model.to(device)

    pbar = tqdm(total=epochs * len(train_loader), desc="Training")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = augment(X_batch).to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix(epoch=epoch + 1, loss=f"{loss.item():.4f}")

        avg_train = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        pbar.set_postfix(
            epoch=epoch + 1,
            train_loss=f"{avg_train:.4f}",
            val_acc=f"{val_acc:.4f}",
        )

    pbar.close()
    return history
