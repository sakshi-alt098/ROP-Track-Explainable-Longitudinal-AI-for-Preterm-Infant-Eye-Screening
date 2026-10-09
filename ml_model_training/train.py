"""
Module 5: Training Pipeline, Model Checkpointing, and Evaluation Metrics
"""

import os
import time
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score

from .config import cfg
from .dataset import RetinalDataset, load_balanced_partitions
from .model import ROPCustomCNN
from .explainability import GradCAMExplainer, render_abnormality_circles

def run_training():
    print("=" * 65)
    print("  LAUNCHING ROP HIGH-ACCURACY TRAINING PIPELINE")
    print("=" * 65)

    dataset_cache = os.path.expanduser(
        r"~/.cache/kagglehub/datasets/jananowakova/retinal-image-dataset-of-infants-and-rop/versions/4/images/images"
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test), cat_to_idx = load_balanced_partitions(dataset_cache)
    idx_to_cat = {i: c for c, i in cat_to_idx.items()}
    idx_to_name = {i: f"{c}: {cfg.CLASS_NAMES[c]}" for i, c in idx_to_cat.items()}

    train_ds = RetinalDataset(X_train, y_train, is_train=True)
    val_ds   = RetinalDataset(X_val, y_val, is_train=False)
    test_ds  = RetinalDataset(X_test, y_test, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ROPCustomCNN(num_classes=len(cfg.CATEGORIES)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights_path = cfg.MODEL_DIR / "rop_custom_cnn_best.pth"

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / train_total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / val_total
        scheduler.step(val_acc)

        print(f"Epoch [{epoch:02d}/{cfg.EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"Elapsed: {time.time()-t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'idx_to_name': idx_to_name,
                'val_acc': val_acc
            }, best_weights_path)

    # Final evaluation on holdout test set
    checkpoint = torch.load(best_weights_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_acc = 100.0 * (all_preds == all_targets).mean()

    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    summary = {
        "test_accuracy": float(test_acc),
        "best_val_accuracy": float(best_val_acc),
        "classes": idx_to_name
    }
    with open(cfg.RESULTS_DIR / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    run_training()
