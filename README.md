# Electron-Photon Classifier

This project implements a deep learning model for classifying electrons and photons using a custom ResNet15 architecture. The model analyzes calorimeter data from particle physics experiments to distinguish between electron and photon signatures.

## Project Overview

The classifier uses a ResNet15 architecture with the following key features:
- Custom residual blocks with batch normalization
- Two input channels (ECAL and HCAL calorimeter data)
- Achieves 73.39% accuracy on test set
- ROC AUC score of 0.80

## Project Structure

```
.
├── models/
│   ├── __init__.py
│   ├── block.py         # Implementation of ResNet building blocks
│   └── models.py        # ResNet15_v1 and ResNet15_v2 architectures
├── train/
│   ├── __init__.py
│   ├── evaluate.py      # Model evaluation utilities
│   └── train.py        # Training loop and scheduler implementations
├── utils/
│   ├── __init__.py
│   ├── data_loader.py   # Dataset loading and preprocessing
│   ├── pre_processing.py # Data normalization
│   └── visualization.py  # Plotting utilities for results
├── main.py             # Training script
├── inference.py        # Inference script
└── main.ipynb         # Interactive notebook for training/testing
```

## Prerequisites
- Python 3.7+
- PyTorch
- h5py
- numpy
- scikit-learn
- matplotlib
- torchsummary

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JDhruvR/ElectronPhotonClassifier.git
cd ElectronPhotonClassifier
```

2. Download the dataset:
```bash
mkdir data
cd data
curl -o SinglePhoton249k.hdf5 https://cernbox.cern.ch/remote.php/dav/public-files/AtBT8y4MiQYFcgc/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5
curl -o SingleElectron249k.hdf5 https://cernbox.cern.ch/remote.php/dav/public-files/FbXw3V4XNyYB3oA/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5
cd ..
```

## Usage

### Training
```bash
python main.py
```

### Inference
```bash
python inference.py
```

For interactive development, you can use `main.ipynb` which contains both training and testing code. This notebook can be run directly in Google Colab.

## Model Architecture

The ResNet15_v2 architecture includes:
- Initial convolution layer with 32 filters
- 5 residual blocks with increasing channel dimensions (32→64→128→256→512→1024)
- Batch normalization and ReLU activation throughout
- Final fully connected layer for binary classification

## Training

The model is trained with:
- Adam optimizer
- Cross-entropy loss
- Learning rate: 1e-3
- Weight decay: 1e-4
- Plateau learning rate scheduler
- 30 epochs

## Results

The model achieves:
- Accuracy: 73.39%
- ROC AUC Score: 0.80

Our implementation replicates the predictions from the [E2E CMS paper](https://arxiv.org/abs/1807.11916).
