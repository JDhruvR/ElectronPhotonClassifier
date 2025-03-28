import torch
import torch.nn as nn

from utils import load_electron_photon_dataset, plot_roc_curve
from models import ResNet15_v2
from train import evaluate_model

electron_dataset_path = "./data/SingleElectron249k.hdf5"
photon_dataset_path = "./data/SinglePhoton249k.hdf5"

train_loader, val_loader, test_loader, transform = load_electron_photon_dataset(
    electron_dataset_path, photon_dataset_path, (0.64, 0.16, 0.20), 512
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = ResNet15_v2().to(device)
model.summary()

model.load_state_dict(torch.load('./saved_models/resnet15_v2.pth', map_location=device))

evaluate_model(model, test_loader, criterion=nn.CrossEntropyLoss(), device=device)
plot_roc_curve(model, test_loader)
