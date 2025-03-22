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
