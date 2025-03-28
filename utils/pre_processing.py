import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

def process_data(dataset:Dataset):
    """
    Calculate the mean and standard deviation of the dataset for normalization

    Args:
        dataset (Dataset): Dataset object containing the data

    Returns:
        Tuple: Tuple containing mean and standard deviation of the dataset
    """
    #Create a temporary DataLoader to iterate through the data
    temp_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    #Initialize variables for calculating mean and std
    channels_sum = torch.zeros(2)  #2 channels in dataset
    channels_squared_sum = torch.zeros(2)
    num_batches = 0

    # Calculate mean and std across all images
    print("Calculating dataset statistics for normalization...")
    for data, _ in tqdm(temp_loader, total=len(temp_loader)):
        # data shape: [batch_size, channels, height, width]
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    # Calculate mean and std
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    print(f"Dataset statistics - Mean: {mean.tolist()}, Std: {std.tolist()}")
    return mean.tolist(), std.tolist()

def create_transform(dataset:Dataset):
    """
    Create normalization transforms for the dataset based on its statistics

    Args:
        dataset (Dataset): Dataset object used to compute normalization statistics

    Returns:
        tuple: A tuple containing two identical normalization transforms
    """
    mean, std = process_data(dataset)
    normalize_transform = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    return normalize_transform, normalize_transform
