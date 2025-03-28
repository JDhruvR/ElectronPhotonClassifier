import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc

def show_image(img_tensor, figsize=(10, 10)):
    """
    Display an image tensor using matplotlib

    Args:
        img_tensor (torch.Tensor): Image tensor of shape (C, H, W), C=2 for ECAL/HCAL channels
        figsize (tuple, optional): Figure size in inches
    """
    plt.figure(figsize=figsize)

    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = np.array(img_tensor)

    # Display first channel (ECAL)
    plt.subplot(1, 2, 1)
    plt.imshow(img[0, :, :], cmap='viridis')
    plt.title('ECAL')
    plt.colorbar()

    # Display second channel (HCAL)
    plt.subplot(1, 2, 2)
    plt.imshow(img[1, :, :], cmap='plasma')
    plt.title('HCAL')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def show_loss(training_loss: list, validation_loss: list):
    """
    Plot training and optional validation loss curves.

    Args:
        training_loss: List of training loss values per epoch
        validation_loss: Optional list of validation loss values per epoch
    """

    # Convert tensors to CPU if they're on GPU
    training_loss = [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in training_loss]
    if validation_loss:
        validation_loss = [loss.cpu().item() if torch.is_tensor(loss) else loss for loss in validation_loss]

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(training_loss) + 1)

    plt.plot(epochs, training_loss, 'b-', label='Training Loss', linewidth=2)
    if validation_loss:
        plt.plot(epochs, validation_loss, 'r--', label='Validation Loss', linewidth=2)
        plt.title('Training Loss', fontsize=14)
    else:
        plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

def show_accuracy(training_accuracy: list, validation_accuracy: list):
    """
    Plot training and optional validation accuracycurves.

    Args:
        training_accuracy: List of training accuracy values per epoch
        validation_accuracy: Optional list of validation accuracy values per epoch
    """

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(training_accuracy) + 1)

    plt.plot(epochs, training_accuracy, 'b-', label='Training Accuracy', linewidth=2)
    if validation_accuracy:
        plt.plot(epochs, validation_accuracy, 'r--', label='Validation Accuracy', linewidth=2)
        plt.title('Training Accuracy', fontsize=14)
    else:
        plt.title('Training and Validation Accuracy', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_roc_curve(model, val_loader, device='cuda'):
    """
    Plot ROC curve for the validation set.

    Args:
        model: The trained neural network model
        val_loader: Validation data loader
        device: Device to run the predictions on
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU instead")

    model.to(device)
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            # Get model predictions
            outputs = model(data)
            # Convert from log probabilities to probabilities
            probs = torch.exp(outputs)
            # We want the probability for class 1 (usually photons in your case)
            y_pred.extend(probs[:, 1].cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
