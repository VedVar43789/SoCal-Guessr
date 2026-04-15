"""Example prediction code for the DSC 140B SoCalGuessr project.

This file is paired with `train.py`, which trains a model and saves its weights to
`model.pt`. This file loads the saved model and uses it to make predictions on the test
set.

The autograder will call the `predict` function defined below. You are free to change
the implementation, but the function signature must stay the same: it must accept a path
to a directory of test images and return a dictionary mapping each filename to a
predicted label as a string (e.g. "Los_Angeles").

"""

import pathlib

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# these must match the values used during training in train.py
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

# when we trained the model, we resized all images to this size before feeding them into
# the model. We need to do the same thing here, since the model's weights were trained
# on images of this size.
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 32


def load_and_transform_image(path):
    """Load an image from disk and apply the same transforms used during training.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a .jpg image file.

    Returns
    -------
    torch.Tensor
        A tensor of shape (1, 3, IMAGE_WIDTH, IMAGE_HEIGHT) ready to be fed into
        the model.

    """
    image = Image.open(path).convert("RGB")
    pipeline = transforms.Compose(
        [
            transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
            transforms.ToTensor(),
        ]
    )
    return pipeline(image).unsqueeze(0)  # add batch dimension


# we must re-define the model architecture here in order to load the saved weights.


class LogisticRegression(nn.Module):
    """A single linear layer — logistic regression on flattened pixels."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)


def predict(test_dir):
    """Load the saved model and predict a label for every image in `test_dir`.

    Parameters
    ----------
    test_dir : pathlib.Path
        Path to a directory containing .jpg test images.

    Returns
    -------
    dict[str, str]
        A dictionary mapping each image filename (e.g. "00001.jpg") to a predicted
        class label (e.g. "Los_Angeles").

    """
    test_dir = pathlib.Path(test_dir)

    # Step 1) load the trained model.
    input_dim = 3 * IMAGE_WIDTH * IMAGE_HEIGHT
    model = LogisticRegression(input_dim, len(CLASSES))
    model.load_state_dict(torch.load("model.pt", weights_only=True))
    model.eval()

    # Step 2) run prediction on every test image.
    predictions = {}
    with torch.no_grad():
        for path in sorted(test_dir.glob("*.jpg")):
            image = load_and_transform_image(path)
            output = model(image)
            predicted_index = output.argmax(dim=1).item()
            predictions[path.name] = CLASSES[predicted_index]

    return predictions


if __name__ == "__main__":
    preds = predict("./testdata")
    print("Predictions:")
    for filename, label in sorted(preds.items()):
        print(f"{filename}: {label}")
