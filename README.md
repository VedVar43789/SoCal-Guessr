# SoCal-Guessr

A deep learning project for predicting which Southern California city an image belongs to.

The model is trained on labeled city images and predicts one of:
`Anaheim`, `Bakersfield`, `Los_Angeles`, `Riverside`, `SLO`, `San_Diego`.

## Project Highlights

- ResNet18 transfer learning (ImageNet pretrained backbone)
- Reproducible train/validation split with fixed seed
- Training and validation metrics logging
- Automatic generation of:
  - `training_curve.png`
  - `confusion_matrix.png`
- Inference pipeline compatible with saved `model.pt` weights

## Repository Structure

```text
SoCalGuessr/
├── train.py               # Trains ResNet18 and saves model weights
├── predict.py             # Loads model.pt and predicts labels for .jpg images
├── evaluate.py            # Evaluates accuracy (validation split or random sample)
├── p1.py                  # Older baseline/reference predictor implementation
├── requirements.txt       # Python dependencies
├── training_curve.png     # Example generated training plot
├── confusion_matrix.png   # Example generated confusion matrix
├── model.pt               # Trained model weights (ignored in git)
├── data/                  # Training image dataset (ignored in git)
└── .venv/                 # Local virtual environment (ignored in git)
```

## Setup

### 1) Clone

```bash
git clone https://github.com/VedVar43789/SoCal-Guessr.git
cd SoCal-Guessr
```

### 2) Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Data Format

`train.py` expects images in `./data` as `.jpg` files named like:

```text
<CityName>-<index>.jpg
```

Example:

```text
Los_Angeles-00001.jpg
San_Diego-01432.jpg
```

The city prefix is used as the training label.

## Training

### Standard training (80/20 split, 15 epochs)

```bash
python train.py
```

### Quick sanity run (2 epochs)

```bash
python train.py --quick
```

### Full-data training (no validation split)

```bash
python train.py --full
```

After training, you should get:

- `model.pt`
- `training_curve.png`
- `confusion_matrix.png` (only when validation is used)

## Evaluation

### Evaluate on validation split

```bash
python evaluate.py
```

### Evaluate on a random sample

```bash
python evaluate.py --sample 100
```

### With custom seed

```bash
python evaluate.py --seed 42
```

## Inference

Use `predict.py` in your own script:

```python
from predict import predict

preds = predict("./some_test_folder")
print(preds["00001.jpg"])
```

Expected return type:

- `dict[str, str]`
- Keys: filenames like `00001.jpg`
- Values: predicted city labels

## Notes

- Ensure `model.pt` exists before running inference/evaluation.
- `predict.py` and `train.py` must stay consistent on:
  - class ordering
  - image size
  - normalization settings
- Large/local artifacts are excluded from Git via `.gitignore`:
  - `data/`
  - `.venv/`
  - `model.pt`

## License

This repository currently includes an MIT `LICENSE` file.