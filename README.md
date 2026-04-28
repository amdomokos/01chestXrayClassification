## Project Overview

This project implements a benchmarking pipeline for evaluating deep learning models on bioimaging datasets. The goal is to standardize preprocessing, training evaluation, and dataset handling to enable consistent comparison across model architectures.

The pipeline is designed with medical imaging use cases in mind, where dataset imbalance, preprocessing consistency, and evaluation reliability are critical.


## Problem Context

Medical imaging datasets often exhibit:
- Strong class imbalance
- Variability in acquisition quality
- Sensitivity to preprocessing choices

This project focuses on building a reproducible benchmarking framework to evaluate how different models perform under these constraints.


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

Dataset loading is handled through PyTorch `ImageFolder`, enabling automatic label inference from directory structure.


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
- Integrate confusion matrix and ROC analysis
- Extend support to additional imaging modalities
- Add experiment tracking (e.g., Weights & Biases or TensorBoard)


## Summary

This project provides a lightweight but structured framework for benchmarking deep learning


## Acknowledgements

Chest X-Ray Images (Pneumonia) dataset, hosted on Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

This project is built for educational and exploratory purposes in medical image classification using deep learning.
