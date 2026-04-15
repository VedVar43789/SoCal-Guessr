"""Training code for the DSC 140B SoCalGuessr project.

Uses transfer learning with ResNet18 pretrained on ImageNet. ResNet18 captures
spatial features (edges, textures, objects) that logistic regression cannot,
which is essential for distinguishing cities by visual cues like architecture,
vegetation, and signage.

Reasoning for key choices:
- ResNet18: Good accuracy/size tradeoff (~44MB), fits under 50MB limit
- 224x224 images: Standard input size for pretrained models; preserves detail
- ImageNet normalization: Pretrained weights expect these mean/std values
- Data augmentation: Reduces overfitting on ~9k images
- Adam optimizer: Adaptive learning rates, works well for transfer learning

Usage:
  python train.py          # Train with 80/20 split (15 epochs)
  python train.py --full   # Train on 100% of data for final submission
  python train.py --quick  # Quick test (2 epochs)
"""

import argparse
import pathlib
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix


# Configuration ------------------------------------------------------------------------

TRAIN_DIR = pathlib.Path("./data")

CLASSES = sorted(
    [
        "Anaheim",
        "Bakersfield",
        "Los_Angeles",
        "Riverside",
        "SLO",
        "San_Diego",
    ]
)
CLASS_TO_NUMBER = {name: i for i, name in enumerate(CLASSES)}

# 224x224 is the standard input size for ImageNet-pretrained models. Smaller sizes
# (like 64x32 in the baseline) lose spatial detail that CNNs need to distinguish
# cities (e.g., road sign styles, vegetation, architecture).
IMAGE_SIZE = 224

# ImageNet normalization: pretrained ResNet expects images normalized with these
# mean and std. Using different values would hurt performance.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 32  # Smaller than baseline to fit ResNet in GPU memory if available
LEARNING_RATE = 1e-4  # Lower than baseline: pretrained weights need gentle updates
EPOCHS = 15
VALIDATION_FRACTION = 0.2
RANDOM_SEED = 42  # Fixed seed so train/val split is reproducible (for unseen testing)
MODEL_SAVE_PATH = "model.pt"
TRAINING_CURVE_PATH = "training_curve.png"
CONFUSION_MATRIX_PATH = "confusion_matrix.png"


# Dataset ------------------------------------------------------------------------------

class SoCalDataset(Dataset):
    """Loads images from the training set with labels from filenames."""

    def __init__(self, root, transform=None, indices=None):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.samples = []
        for path in sorted(self.root.glob("*.jpg")):
            label = path.name.rsplit("-", 1)[0]
            self.samples.append((path, CLASS_TO_NUMBER[label]))
        self.indices = indices  # If set, only use these indices (for train/val split)

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.samples)

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        path, label = self.samples[real_idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Model --------------------------------------------------------------------------------

def create_model(num_classes):
    """Create ResNet18 with pretrained backbone and custom classifier head.

    ResNet18 has 512-dimensional features from the last conv layer. We replace
    the original 1000-class head with a 6-class head for our cities.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace final fully-connected layer: 512 features -> 6 classes
    model.fc = nn.Linear(512, num_classes)
    return model


# Training -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run 2 epochs for quick testing")
    parser.add_argument("--full", action="store_true", help="Train on 100%% of data (no validation split)")
    args = parser.parse_args()
    epochs = 2 if args.quick else EPOCHS
    use_full_data = args.full
    if args.quick:
        print("Quick mode: 2 epochs only")
    if use_full_data:
        print("Full data mode: training on 100% of dataset (no validation split)")

    start_time = time.time()

    # Training transforms: augmentation helps with limited data (~9k images).
    # RandomHorizontalFlip: street views are often symmetric; flip doesn't change city.
    # ColorJitter: lighting/weather varies; makes model robust to these variations.
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Validation: no augmentation, only resize and normalize.
    val_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    full_dataset = SoCalDataset(TRAIN_DIR, transform=None)
    if use_full_data:
        train_dataset = SoCalDataset(TRAIN_DIR, transform=train_transform, indices=None)
        val_loader = None
    else:
        val_size = int(len(full_dataset) * VALIDATION_FRACTION)
        train_size = len(full_dataset) - val_size
        torch.manual_seed(RANDOM_SEED)
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
        train_dataset = SoCalDataset(
            TRAIN_DIR, transform=train_transform, indices=list(train_subset.indices)
        )
        val_dataset = SoCalDataset(
            TRAIN_DIR, transform=val_transform, indices=list(val_subset.indices)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = create_model(len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Record training curve for the report
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=True,
        )
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

            # Update progress bar with current metrics
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.4f}",
            )

        avg_loss = total_loss / total
        train_losses.append(avg_loss)
        accuracy = correct / total

        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    val_total += images.size(0)
            val_accuracy = val_correct / val_total
            val_accuracies.append(val_accuracy)
            model.train()
            print(
                f"\n  Epoch {epoch + 1}/{epochs} complete  |  "
                f"loss: {avg_loss:.4f}  |  "
                f"train_acc: {accuracy:.4f}  |  "
                f"val_acc: {val_accuracy:.4f}\n"
            )
        else:
            print(
                f"\n  Epoch {epoch + 1}/{epochs} complete  |  "
                f"loss: {avg_loss:.4f}  |  "
                f"train_acc: {accuracy:.4f}\n"
            )

    # Save state_dict (not full model) for portability. predict.py will define
    # the same architecture and load these weights.
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Saved model to {MODEL_SAVE_PATH}")

    # --- Plots for report (PDF requirements) ---

    # 1. Training curve: iteration (epoch) vs empirical risk (required by PDF)
    x_epochs = range(1, epochs + 1)
    if val_loader is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        ax1.plot(x_epochs, train_losses, "b-o", markersize=6)
        ax1.set_ylabel("Empirical Risk (Cross-Entropy Loss)")
        ax1.set_title("Training Curve: Epoch vs Empirical Risk")
        ax1.grid(True, alpha=0.3)
        ax2.plot(x_epochs, val_accuracies, "g-o", markersize=6)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation Accuracy")
        ax2.set_title("Validation Accuracy Over Training")
        ax2.grid(True, alpha=0.3)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
        ax1.plot(x_epochs, train_losses, "b-o", markersize=6)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Empirical Risk (Cross-Entropy Loss)")
        ax1.set_title("Training Curve: Epoch vs Empirical Risk (100% data)")
        ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(TRAINING_CURVE_PATH)
    plt.close()
    print(f"Saved training curve to {TRAINING_CURVE_PATH}")

    # 2. Confusion matrix (only when we have validation set)
    if val_loader is not None:
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(CLASSES)),
            yticks=np.arange(len(CLASSES)),
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            xlabel="Predicted",
            ylabel="True",
            title="Model Confusion Matrix (Validation Set)",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(
                    j, i, cm[i, j],
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=10,
                )
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_PATH)
        plt.close()
        print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")

    elapsed = (time.time() - start_time) / 60
    print(f"\nTraining completed in {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()
