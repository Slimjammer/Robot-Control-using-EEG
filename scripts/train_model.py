"""
scripts/train_model.py
======================
Re-train the EEGModel on BCI-2a training files using the src/ modules.

Usage:
  python scripts/train_model.py
  python scripts/train_model.py --epochs 80 --output output/my_model.pth
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.config.settings import (
    DATA_DIR,
    OUTPUT_DIR,
    SAMPLING_RATE,
    N_CLASSES_CHECKPOINT,
    ACTIVE_CLASSES,
)
from src.data.loader        import load_bci2a_data
from src.preprocessing.pipeline import preprocess_raw
from src.model.eeg_model    import EEGModel
from src.training.trainer   import train, evaluate
from src.training.metrics   import final_evaluation, plot_history


def parse_args():
    p = argparse.ArgumentParser(description="Train EEGModel on BCI-2a")
    p.add_argument("--epochs",  type=int,   default=80)
    p.add_argument("--batch",   type=int,   default=32)
    p.add_argument("--lr",      type=float, default=1e-4)
    p.add_argument("--output",  type=str,   default=str(OUTPUT_DIR / "eeg_model_demo.pth"))
    p.add_argument("--no-ica",  action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Train] Device: {device}  |  Epochs: {args.epochs}  |  Batch: {args.batch}")

    # ── Load data ─────────────────────────────────────────────────────────────
    raws = load_bci2a_data(str(DATA_DIR), file_suffix="T.gdf")

    # ── Preprocess all subjects ───────────────────────────────────────────────
    X_all, y_all = [], []
    for i, raw in enumerate(raws):
        print(f"[Train] Preprocessing subject {i+1}/{len(raws)} …")
        X, y = preprocess_raw(raw)
        X_all.append(X)
        y_all.append(y)

    X = np.concatenate(X_all)   # (N, n_channels, T)
    y = np.concatenate(y_all)   # (N,)

    print(f"[Train] Total epochs: {len(X)}  |  Classes: {np.unique(y)}")

    # ── Train / val split ─────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Normalize (channel-wise z-score over training set) ────────────────────
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std  = X_train.std(axis=(0, 2),  keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std

    # ── Convert to (N, T, C) for the model ────────────────────────────────────
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val   = np.transpose(X_val,   (0, 2, 1))

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=args.batch, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        ),
        batch_size=args.batch, shuffle=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    n_channels = X_train.shape[2]
    model      = EEGModel(n_channels=n_channels, n_classes=N_CLASSES_CHECKPOINT)
    criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"[Train] n_channels={n_channels}, n_classes={N_CLASSES_CHECKPOINT} "
          f"(active: {ACTIVE_CLASSES})")

    # ── Train ──────────────────────────────────────────────────────────────────
    history = train(model, train_loader, val_loader, optimizer, criterion,
                    scheduler=scheduler, epochs=args.epochs, device=device)

    plot_history(history, title_prefix="BCI-2a")

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), args.output)
    print(f"[Train] Model saved → {args.output}")

    # ── Final eval ────────────────────────────────────────────────────────────
    final_evaluation(model, val_loader, device=device)


if __name__ == "__main__":
    main()
