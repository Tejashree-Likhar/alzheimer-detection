"""
Data loading and preprocessing for the Alzheimer MRI Disease Classification dataset.

Source: https://huggingface.co/datasets/Falah/Alzheimer_MRI
Classes:
    0 - NonDemented
    1 - VeryMildDemented
    2 - MildDemented
    3 - ModerateDemented
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224  # standard input size for ResNet/EfficientNet backbones

# ImageNet normalization stats — used because we fine-tune from ImageNet-pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool = True):
    """Return the torchvision transform pipeline for train or eval mode."""
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # MRI scans are grayscale
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class AlzheimerMRIDataset(Dataset):
    """Wraps a Hugging Face dataset split as a PyTorch Dataset."""

    def __init__(self, hf_split, transform=None):
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        example = self.hf_split[idx]
        image = example["image"].convert("RGB")
        label = example["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_alzheimer_dataloaders(batch_size: int = 32, num_workers: int = 2, val_split: float = 0.1):
    """
    Downloads (and caches) the Falah/Alzheimer_MRI dataset from the Hugging Face Hub
    and returns train / val / test DataLoaders.

    A slice of the original training split is carved out as a validation set since the
    source dataset only ships train/test splits.
    """
    raw = load_dataset("Falah/Alzheimer_MRI")

    full_train = raw["train"]
    test_split = raw["test"]

    # Carve out a validation split from the training data (stratification not required
    # here since the dataset is already reasonably balanced per class within splits)
    split = full_train.train_test_split(test_size=val_split, seed=42)
    train_split, val_split_ds = split["train"], split["test"]

    train_ds = AlzheimerMRIDataset(train_split, transform=get_transforms(train=True))
    val_ds = AlzheimerMRIDataset(val_split_ds, transform=get_transforms(train=False))
    test_ds = AlzheimerMRIDataset(test_split, transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick smoke test: python -m src.data
    train_loader, val_loader, test_loader = load_alzheimer_dataloaders(batch_size=8)
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, labels: {labels}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
