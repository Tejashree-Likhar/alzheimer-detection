"""
Model architecture for Alzheimer's MRI classification.

Uses transfer learning from an ImageNet-pretrained ResNet18 backbone, which works
well on datasets of this size (a few thousand images) since training a CNN from
scratch would likely overfit.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from src.data import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = False) -> nn.Module:
    """
    Builds a ResNet18 with its final fully-connected layer replaced for
    Alzheimer's stage classification.

    Args:
        num_classes: number of output classes.
        freeze_backbone: if True, freezes all convolutional layers and only
            trains the new classification head (faster, useful for small datasets
            or quick experimentation).
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )
    return model


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # expected: [2, 4]
