import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def evaluate_model(model:nn.Module, dataloader:DataLoader, criterion:nn.Module, device:str='cpu', dataset_name:str='validation'):
    """
    Evaluate model over the data in Dataloader.

    Args:
        model: Neural network model
        dataloader: Dataloader for the data we want to evaluate our model on
        criterion: The criterion to calculate our evaluation metrics
        device: Device to use to evaluate the model

    Returns:
        avg_loss (float): Average loss over the dataset given
        accuracy (float): Accuracy over the dataset given
    """
    total_loss = 0.0
    correct = 0
    total = 0

    original_device = next(model.parameters()).device
    model.to(torch.device(device))
    model.eval()
    with torch.no_grad():
        for data, label in tqdm(dataloader, desc='evaluating'):
            #Move data to device
            data, label = data.to(torch.device(device)), label.to(torch.device(device))
            #Forward pass
            output = model(data)
            loss = criterion(output, label)
            #Accumulate loss
            total_loss += loss
            #Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        # Clear some memory after each batch
        if device == 'cuda':
            torch.cuda.empty_cache()

    model.to(original_device)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    print(f'Loss in {dataset_name} : {avg_loss}, Accuracy : {accuracy}')

    return avg_loss, accuracy
