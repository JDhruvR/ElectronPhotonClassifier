import matplotlib.pyplot as plt
import torch
import numpy as np

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
