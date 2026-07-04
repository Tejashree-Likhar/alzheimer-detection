"""
Evaluation script: loads a trained checkpoint and reports detailed metrics
(confusion matrix, per-class precision/recall/F1) on the held-out test split.

Usage:
    python -m src.evaluate --checkpoint alzheimer_model.pt
"""

import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from src.data import load_alzheimer_dataloaders, CLASS_NAMES
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Alzheimer's classifier")
    parser.add_argument("--checkpoint", type=str, default="alzheimer_model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-confusion-matrix", type=str, default="confusion_matrix.png")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, _, test_loader = load_alzheimer_dataloaders(batch_size=args.batch_size)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("Classification report:\n")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(args.save_confusion_matrix, dpi=150)
    print(f"Confusion matrix saved to {args.save_confusion_matrix}")


if __name__ == "__main__":
    main()
