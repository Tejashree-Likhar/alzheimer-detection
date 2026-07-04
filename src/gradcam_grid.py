"""
Generates a grid of Grad-CAM visualizations, one sample per class, useful for
a README screenshot or a model-report figure.

Usage:
    python -m src.gradcam_grid --checkpoint alzheimer_model.pt --output gradcam_grid.png
"""

import argparse

import matplotlib.pyplot as plt
import torch

from src.data import load_alzheimer_dataloaders, get_transforms, CLASS_NAMES
from src.model import build_model
from src.gradcam import GradCAM, overlay_heatmap_on_image


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM sample grid")
    parser.add_argument("--checkpoint", type=str, default="alzheimer_model.pt")
    parser.add_argument("--output", type=str, default="gradcam_grid.png")
    parser.add_argument("--samples-per-class", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _, _, test_loader = load_alzheimer_dataloaders(batch_size=1)
    test_dataset = test_loader.dataset

    # Collect one (or more) example index per class from the raw HF split
    class_to_indices = {i: [] for i in range(len(CLASS_NAMES))}
    for idx in range(len(test_dataset.hf_split)):
        label = test_dataset.hf_split[idx]["label"]
        if len(class_to_indices[label]) < args.samples_per_class:
            class_to_indices[label].append(idx)
        if all(len(v) >= args.samples_per_class for v in class_to_indices.values()):
            break

    transform = get_transforms(train=False)
    cam = GradCAM(model, target_layer=model.layer4)

    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(args.samples_per_class, n_classes * 2,
                              figsize=(4 * n_classes, 4 * args.samples_per_class))
    if args.samples_per_class == 1:
        axes = axes.reshape(1, -1)

    for class_idx, indices in class_to_indices.items():
        for row, idx in enumerate(indices):
            example = test_dataset.hf_split[idx]
            original_image = example["image"].convert("RGB")
            input_tensor = transform(original_image).unsqueeze(0).to(device)

            heatmap, pred_idx, probs = cam.generate(input_tensor)
            overlay = overlay_heatmap_on_image(heatmap, original_image)

            col_orig = class_idx * 2
            col_cam = class_idx * 2 + 1

            axes[row, col_orig].imshow(original_image)
            axes[row, col_orig].set_title(f"True: {CLASS_NAMES[class_idx]}", fontsize=9)
            axes[row, col_orig].axis("off")

            correct = "✓" if pred_idx == class_idx else "✗"
            axes[row, col_cam].imshow(overlay)
            axes[row, col_cam].set_title(
                f"Grad-CAM | Pred: {CLASS_NAMES[pred_idx]} {correct}", fontsize=9
            )
            axes[row, col_cam].axis("off")

    cam.remove_hooks()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved Grad-CAM grid to {args.output}")


if __name__ == "__main__":
    main()
