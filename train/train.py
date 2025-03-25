import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluate import evaluate_model
from utils.visualization import show_loss, show_accuracy

def train(model:nn.Module, epochs:int, criterion:nn.Module, train_loader:DataLoader, val_loader:DataLoader, device='cuda', lr:float=1e-4, scheduler_type=None):
    """
    Trains the given model using the dataset provided by the dataloader using Adam and given criterion, lr, and epochs on the given device type. Uses validation dataset to also track validation metrics. Plots the metrics calculated using plt.

    Args:
        model (torch.nn.Module) : The model to train
        criterion (torch.nn.Module) : The loss function to calculate the gradients
        train_loader, val_loader (torch.utils.data.DataLoader) : Training, Validation datasets to use to train the model
        lr (float) : Learning Rate for the Optimizer
        epochs (float) : Number of epochs to train the model for
        device (str) : cuda or cpu. Device to train the model on
        scheduler_type (str): Type of scheduler to use ('constant', 'step', 'cosine', 'plateau')
    """

    model.to(torch.device(device))
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Create the appropriate scheduler
    scheduler = create_scheduler(optimizer, scheduler_type, epochs)

    training_loss=[]
    training_accuracy=[]
    eval_loss=[]
    eval_accuracy=[]

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()

        for data, label in tqdm(train_loader, desc=f'Epoch : {epoch+1} / {epochs}'):
            #Move data to the same device the model is on.
            data, label = data.to(torch.device(device)), label.to(torch.device(device))
            #Forward pass
            output = model(data)
            loss = criterion(output.squeeze(), label)
            #Backprop
            optimizer.zero_grad()
            loss.backward()
            #Update weights
            optimizer.step()
            #Update statistics
            _, predicted = torch.max(output, 1)
            running_loss += loss.item()
            total += label.size(0)
            correct += (label == predicted).sum().item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(running_loss)
        else:
            # Other schedulers just need a step() call
            scheduler.step()

        #Training Statistics
        avg_loss = running_loss / len(train_loader.dataset)
        accuracy = (correct / total) * 100

        training_loss.append(avg_loss)
        training_accuracy.append(accuracy)

        #Display Training Statistics
        print(
            f"total_loss: {running_loss}, average loss: {avg_loss:.4f}, accuracy: {accuracy:.2f}, for epoch: {epoch + 1}"
        )

        #Validation
        loss, accuracy = evaluate_model(model, val_loader, criterion, device=device)

        eval_loss.append(loss)
        eval_accuracy.append(accuracy)

    show_loss(training_loss, eval_loss)
    show_accuracy(training_accuracy, eval_accuracy)

def create_scheduler(optimizer, scheduler_type, epochs):
    """
    Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to wrap with a scheduler
        scheduler_type: Type of scheduler ('step', 'cosine', 'plateau', 'onecycle')
        epochs: Number of epochs for training

    Returns:
        A learning rate scheduler
    """
    if scheduler_type == 'constant' or scheduler_type is None:
        # Official PyTorch constant learning rate scheduler
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)

    if scheduler_type == 'step':
        # Step decay: reduce learning rate by a factor of 0.1 every 3 epochs
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    elif scheduler_type == 'cosine':
        # Cosine annealing: gradually reduce learning rate following a cosine curve
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    elif scheduler_type == 'plateau':
        # Reduce on plateau: reduce learning rate when validation loss plateaus
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=2,
            threshold=0.01, threshold_mode='rel', cooldown=0
        )

    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
