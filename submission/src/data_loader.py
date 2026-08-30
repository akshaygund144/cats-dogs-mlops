from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

IMAGE_SIZE = 128
BATCH_SIZE = 64

# Windows CPU optimization.
# Using 0 workers avoids multiprocessing overhead and
# potential DataLoader issues on Windows.
NUM_WORKERS = 0

# CPU training does not require pinned memory.
PIN_MEMORY = False


# =========================================================
# Image Normalization
# =========================================================

# Standard ImageNet normalization.
# This also keeps the pipeline compatible with
# torchvision models if transfer learning is used later.

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# =========================================================
# Training Transformations
# =========================================================

# Data augmentation is intentionally retained because
# it is required by the assignment.

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# =========================================================
# Validation / Test Transformations
# =========================================================

# Validation and test data remain deterministic.
# No random augmentation is applied.

eval_transforms = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# =========================================================
# Create Datasets
# =========================================================

def create_datasets():

    train_dataset = datasets.ImageFolder(
        PROCESSED_DIR / "train",
        transform=train_transforms
    )

    validation_dataset = datasets.ImageFolder(
        PROCESSED_DIR / "validation",
        transform=eval_transforms
    )

    test_dataset = datasets.ImageFolder(
        PROCESSED_DIR / "test",
        transform=eval_transforms
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset
    )


# =========================================================
# Create DataLoaders
# =========================================================

def create_dataloaders():

    (
        train_dataset,
        validation_dataset,
        test_dataset
    ) = create_datasets()

    # Training DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Validation DataLoader
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Test DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return (
        train_loader,
        validation_loader,
        test_loader
    )


# =========================================================
# Quick Verification
# =========================================================

if __name__ == "__main__":

    (
        train_loader,
        validation_loader,
        test_loader
    ) = create_dataloaders()

    print("=" * 60)
    print("Cats vs Dogs DataLoader Verification")
    print("=" * 60)

    print(
        "Classes:",
        train_loader.dataset.classes
    )

    print(
        "Training images:",
        len(train_loader.dataset)
    )

    print(
        "Validation images:",
        len(validation_loader.dataset)
    )

    print(
        "Test images:",
        len(test_loader.dataset)
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Image size:",
        IMAGE_SIZE
    )

    print(
        "DataLoader workers:",
        NUM_WORKERS
    )

    print(
        "Pin memory:",
        PIN_MEMORY
    )

    images, labels = next(iter(train_loader))

    print(
        "Batch image shape:",
        images.shape
    )

    print(
        "Batch label shape:",
        labels.shape
    )

    print("=" * 60)
    print("DataLoader verification completed successfully.")
    print("=" * 60)