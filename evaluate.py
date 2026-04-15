"""Evaluate model accuracy on the data.

Since the test set is withheld, validation accuracy is the best proxy for test performance.
Uses the same 80/20 split logic as train.py (with fixed seed for reproducibility).

Usage:
  python evaluate.py           # Full validation split (~1836 images)
  python evaluate.py --sample 100   # Random sample of 100 images
"""

import argparse
import pathlib
import random

import torch

# Must match train.py
CLASSES = sorted([
    "Anaheim", "Bakersfield", "Los_Angeles", "Riverside", "SLO", "San_Diego",
])
CLASS_TO_NUMBER = {name: i for i, name in enumerate(CLASSES)}

# Import predict to reuse model loading
from predict import predict, MODEL_PATH


def evaluate(data_dir="./data", val_fraction=0.2, seed=42):
    """Compute accuracy on validation split. Returns (accuracy, total_correct, total)."""
    data_dir = pathlib.Path(data_dir)

    # Build list of (path, true_label) for all images
    samples = []
    for path in sorted(data_dir.glob("*.jpg")):
        true_city = path.name.rsplit("-", 1)[0]
        samples.append((path, CLASS_TO_NUMBER[true_city]))

    # Same 80/20 split as train.py (with fixed seed)
    torch.manual_seed(seed)
    n_val = int(len(samples) * val_fraction)
    n_train = len(samples) - n_val

    class IndexDataset(torch.utils.data.Dataset):
        def __init__(self, n):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, i):
            return i

    _, val_subset = torch.utils.data.random_split(
        IndexDataset(len(samples)), [n_train, n_val]
    )
    val_indices = list(val_subset.indices)

    # Create a temp dir with only val images, or predict in place
    # predict() needs a directory - we'll run on full data and filter
    preds = predict(data_dir)

    correct = 0
    total = 0
    for idx in val_indices:
        path, true_label = samples[idx]
        pred_city = preds[path.name]
        pred_label = CLASS_TO_NUMBER[pred_city]
        if pred_label == true_label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def evaluate_sample(data_dir="./data", n=100, seed=42):
    """Compute accuracy on n randomly chosen images. Returns (accuracy, correct, total)."""
    data_dir = pathlib.Path(data_dir)

    samples = []
    for path in sorted(data_dir.glob("*.jpg")):
        true_city = path.name.rsplit("-", 1)[0]
        samples.append((path, CLASS_TO_NUMBER[true_city]))

    random.seed(seed)
    n = min(n, len(samples))
    chosen = random.sample(range(len(samples)), n)

    preds = predict(data_dir)

    correct = 0
    for idx in chosen:
        path, true_label = samples[idx]
        pred_city = preds[path.name]
        pred_label = CLASS_TO_NUMBER[pred_city]
        if pred_label == true_label:
            correct += 1

    accuracy = correct / n if n > 0 else 0
    return accuracy, correct, n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Use random sample of N images instead of full validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.sample is not None:
        acc, correct, total = evaluate_sample(n=args.sample, seed=args.seed)
        print(f"Accuracy on {total} randomly chosen images: {correct}/{total} = {acc:.2%}")
    else:
        acc, correct, total = evaluate(seed=args.seed)
        print(f"Validation accuracy: {correct}/{total} = {acc:.2%}")
