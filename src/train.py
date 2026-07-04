"""
Training script for the Alzheimer's MRI classifier.

Usage:
    python -m src.train --epochs 15 --batch-size 32 --lr 3e-4
"""

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import load_alzheimer_dataloaders, CLASS_NAMES
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train an Alzheimer's MRI classifier")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true",
                         help="Freeze convolutional layers and only train the classifier head")
    parser.add_argument("--output", type=str, default="alzheimer_model.pt",
                         help="Path to save the best model checkpoint")
    parser.add_argument("--patience", type=int, default=5,
                         help="Early stopping patience (epochs without val improvement)")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset (downloads from Hugging Face Hub on first run)...")
    train_loader, val_loader, test_loader = load_alzheimer_dataloaders(batch_size=args.batch_size)

    model = build_model(freeze_backbone=args.freeze_backbone).to(device)

    # Class weights help counteract the dataset's imbalance (ModerateDemented is rare)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Restore best weights before final test evaluation
    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"\nBest val_acc={best_val_acc:.4f} | Final test_acc={test_acc:.4f}")

    output_path = Path(args.output)
    torch.save({
        "model_state_dict": best_state,
        "class_names": CLASS_NAMES,
        "test_accuracy": test_acc,
    }, output_path)
    print(f"Saved best model checkpoint to {output_path.resolve()}")


if __name__ == "__main__":
    main()
