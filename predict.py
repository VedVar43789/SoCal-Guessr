"""Prediction code for the DSC 140B SoCalGuessr project.

Loads the trained ResNet18 model and predicts city labels for test images.
Transforms must match training exactly: 224x224 resize + ImageNet normalization.
"""

import pathlib
import random

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# Must match train.py exactly
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

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_PATH = "model.pt"
VALIDATION_FRACTION = 0.2
RANDOM_SEED = 42  # Must match train.py for unseen test to use same held-out split


def create_model(num_classes):
    """Same architecture as train.py: ResNet18 with 6-class head."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, num_classes)
    return model


def load_and_transform_image(path):
    """Load image and apply same transforms as training (no augmentation)."""
    image = Image.open(path).convert("RGB")
    pipeline = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return pipeline(image).unsqueeze(0)


def predict(test_dir):
    """Predict city for every image in test_dir.

    Parameters
    ----------
    test_dir : str or pathlib.Path
        Path to directory containing .jpg test images.

    Returns
    -------
    dict[str, str]
        Maps each filename (e.g. "00001.jpg") to predicted city (e.g. "Los_Angeles").
    """
    test_dir = pathlib.Path(test_dir)

    model = create_model(len(CLASSES))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    predictions = {}
    with torch.no_grad():
        for path in sorted(test_dir.glob("*.jpg")):
            image = load_and_transform_image(path)
            output = model(image)
            predicted_index = output.argmax(dim=1).item()
            predictions[path.name] = CLASSES[predicted_index]

    return predictions


# def _get_unseen_indices(data_dir):
#     """Return indices of validation (unseen) images using same 80/20 split as train.py."""
#     data_dir = pathlib.Path(data_dir)
#     all_paths = sorted(data_dir.glob("*.jpg"))
#     n = len(all_paths)
#     n_val = int(n * VALIDATION_FRACTION)
#     n_train = n - n_val

#     class IndexDataset:
#         def __init__(self, n):
#             self.n = n
#         def __len__(self):
#             return self.n
#         def __getitem__(self, i):
#             return i

#     torch.manual_seed(RANDOM_SEED)
#     _, val_subset = torch.utils.data.random_split(
#         IndexDataset(n), [n_train, n_val]
#     )
#     return list(val_subset.indices), all_paths


# if __name__ == "__main__":
#     data_dir = pathlib.Path("./data")
#     preds = predict(data_dir)

#     # Test on UNSEEN data (validation split - images model never saw during training)
#     unseen_indices, all_paths = _get_unseen_indices(data_dir)
#     correct = 0
#     for idx in unseen_indices:
#         path = all_paths[idx]
#         true_city = path.name.rsplit("-", 1)[0]
#         pred_city = preds[path.name]
#         if pred_city == true_city:
#             correct += 1

#     n_unseen = len(unseen_indices)
#     print(f"Accuracy on {n_unseen} UNSEEN images (validation split): {correct}/{n_unseen} = {correct/n_unseen:.2%}")
