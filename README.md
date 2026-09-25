## Project Overview

This project implements a benchmarking pipeline for evaluating deep learning models on bioimaging datasets. The goal is to standardize preprocessing, training evaluation, and dataset handling to enable consistent comparison across model architectures.

The pipeline is designed with medical imaging use cases in mind, where dataset imbalance, preprocessing consistency, and evaluation reliability are critical.


## Problem Context

Medical imaging datasets often exhibit:
- Strong class imbalance
- Variability in acquisition quality
- Sensitivity to preprocessing choices

This project focuses on building a reproducible benchmarking framework to evaluate how different models perform under these constraints.


## Results (held-out test set, n=624)

The original Kaggle validation split (16 images) was unusable, so the original train+val images were pooled and re-split with a stratified, **patient-grouped** split (`python -m src.splits` -> `data/splits.csv`; train 4,360 / val 872 / test 624, no patient in more than one split). The original test set is untouched. Model selection (best val ROC-AUC) uses the validation split only; the test set is evaluated once.

| Metric | Value (95% bootstrap CI) |
|---|---|
| ROC-AUC | 0.962 (0.948-0.974) |
| Sensitivity (PNEUMONIA recall) | 97.4% (95.6-99.0) |
| Specificity (NORMAL recall) | 71.8% (66.2-77.4) |
| Accuracy | 87.8% (85.4-90.2) |

Confusion matrix: TN=168 FP=66 FN=10 TP=380. Class imbalance in the training split is 2.9:1 (PNEUMONIA:NORMAL); the test set is 1.7:1. Grad-CAM is implemented from scratch (forward/backward hooks on `layer4[-1]`) in `notebooks/04_model_playground.ipynb`.

## Reproducing

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu   # Intel GPU (XPU); CPU/CUDA also work
python -m src.splits      # build data/splits.csv
python -m src.train --device auto --amp
python -m src.evaluate
```

`--device auto` picks XPU > CUDA > CPU. Trained on an Intel Arc 140V (bf16 autocast).


## Dataset Structure

The pipeline expects datasets organized in the following format:

```data/
├── train/
│   ├── CLASS_0/
│   └── CLASS_1/
├── val/
│   ├── CLASS_0/
│   └── CLASS_1/
└── test/
    ├── CLASS_0/
    └── CLASS_1/
```

Images stay in the original Kaggle folders; `data/splits.csv` assigns each image to train/val/test, and `src/data_loader.py` reads it (`ManifestDataset`).


## Pipeline Design

### Preprocessing

All images are standardized using:

- Resizing to a fixed resolution (e.g. 224×224)
- Tensor conversion
- Normalization using ImageNet statistics
- Optional augmentation applied only to training data


### Data Handling

To address class imbalance, the pipeline supports:

- Inverse-frequency class weighting
- Weighted random sampling for balanced batch construction

This ensures that minority classes are adequately represented during training.


### Data Inspection Tools

The project includes utilities for:

- Visualizing sample images from each class
- Inspecting batch shapes and normalization effects
- Verifying dataset integrity before training

These checks are used to validate preprocessing correctness and avoid silent data issues.


## Benchmarking Workflow

The standard workflow includes:

1. Dataset loading and validation
2. Preprocessing and augmentation setup
3. Model training on balanced data loaders
4. Evaluation on held-out test set

## Evaluation Metrics

Typical evaluation includes:
- Accuracy
- Class-wise performance breakdown
- Sensitivity to class imbalance
- Visual inspection of predictions (optional extension)


## Implementation Details

The project is implemented in Python using PyTorch. Core components include:

- `torchvision` for dataset handling and transforms
- `DataLoader` for batching and sampling
- `matplotlib` for visualization
- `numpy` for class distribution analysis


## Future Improvements

- Add standardized benchmark suite across multiple architectures
- Extend support to additional imaging modalities
- Add experiment tracking (e.g., Weights & Biases or TensorBoard)


## Summary

This project provides a lightweight but structured framework for benchmarking deep learning models on bioimaging datasets.


## Acknowledgements

Chest X-Ray Images (Pneumonia) dataset, hosted on Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

This project is built for educational and exploratory purposes in medical image classification using deep learning.
