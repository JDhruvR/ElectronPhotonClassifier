import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Dataset, random_split
from .pre_processing import create_transform

def _load_hdf5_to_dataset(file_path: str):
    """
    Load the data in HDF5 file into TensorDataset object

    Args:
        file_path (str) : Path to the HDF5 file

    Returns:
        dataset (torch.utils.data.TensorDataset) : TensorDataset object containing the data in the given HDF5 file
    """
    # Open the HDF5 file
    with h5py.File(file_path, "r") as f:
        # Load image data and labels into numpy arrays
        X_np = np.array(f["X"])
        y_np = np.array(f["y"])

    # Convert to torch tensors
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    # Reshape X from (N, H, W, C) to PyTorch's expected (N, C, H, W) format
    X = X.permute(0, 3, 1, 2)

    dataset = TensorDataset(X, y)

    return dataset


class TransformWrapper(Dataset):
    """
    A simple wrapper that applies transforms to a dataset.
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def load_electron_photon_dataset(
    electron_file_path: str,
    photon_file_path: str,
    train_val_test_split: tuple = (0.64, 0.16, 0.20),
    batch_size: int = 32,
):
    """
    Loads the electron-photon data in HDF5 files, splits them into training, validation, and test set, merges them and returns DataLoader objects of given batch size with the given transforms applied.

    Args:
        electron_file_path (str) : Path to the HDF5 file containing electron data
        photon_file_path (str) : Path to the HDF5 file containing photon data
        train_val_test (tuple) : Tuple containing splitting ratio for the training, validation, and test data. The numbers must be positive and their sum should be equal to 1.
        batch_size (int) : Batch size for the Dataloader

    Returns:
        Tuple of Dataloaders : Training Dataloader, Validation Dataloader, Testing Dataloader
        Transform for Inference (torchvision.transforms.Compose)
    """
    # Load Datasets
    electron_dataset = _load_hdf5_to_dataset(electron_file_path)
    photon_dataset = _load_hdf5_to_dataset(photon_file_path)
    # Enforce constraints for the Splitting ratios
    if len(train_val_test_split) != 3:
        raise ValueError("train_val_test_split expects exactly 3 floats")
    elif (
        sum(train_val_test_split) != 1
        or train_val_test_split[0] < 0
        or train_val_test_split[1] < 0
        or train_val_test_split[2] < 0
    ):
        raise ValueError("train_val_test_split expects each float to be positive and sum to 1")

    # Calculate Split sizes and Split
    electron_train_size = int(len(electron_dataset) * train_val_test_split[0])
    electron_val_size = int(len(electron_dataset) * train_val_test_split[1])
    electron_test_size = len(electron_dataset) - electron_train_size - electron_val_size

    photon_train_size = int(len(photon_dataset) * train_val_test_split[0])
    photon_val_size = int(len(photon_dataset) * train_val_test_split[1])
    photon_test_size = len(photon_dataset) - photon_train_size - photon_val_size

    electron_train_dataset, electron_val_dataset, electron_test_dataset = random_split(
        electron_dataset,
        [electron_train_size, electron_val_size, electron_test_size],
        generator=torch.Generator().manual_seed(42),  # For reproducibility
    )

    photon_train_dataset, photon_val_dataset, photon_test_dataset = random_split(
        photon_dataset,
        [photon_train_size, photon_val_size, photon_test_size],
        generator=torch.Generator().manual_seed(42),  # For reproducibility
    )

    #Merge electron and photon data to make train/val/test datasets
    train_dataset = ConcatDataset([electron_train_dataset, photon_train_dataset])
    val_dataset = ConcatDataset([electron_val_dataset, photon_val_dataset])
    test_dataset = ConcatDataset([electron_test_dataset, photon_test_dataset])

    #Get appropriate transforms for the datasets from Pre-processing
    train_transform, val_transform = create_transform(train_dataset)

    #Apply appropriate transforms to the datasets
    train_dataset = TransformWrapper(train_dataset, transform=train_transform)
    val_dataset = TransformWrapper(val_dataset, transform=val_transform)
    test_dataset = TransformWrapper(test_dataset, transform=val_transform)

    #Create DataLoaders using transformed datasets
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),  # For reproducibility
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_dataloader, val_dataloader, test_dataloader, val_transform
