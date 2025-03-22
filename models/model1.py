import torch.nn as nn
from torchsummary import summary

from .block import Block

class ResNet15_v1(nn.Module):
    def __init__(self, num_classes:int=2):
        super().__init__()

        self.conv1 = nn.ModuleList([
            nn.Conv2d(
                in_channels=2,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(
                num_features=32,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3, stride=1, padding=1
            )
        ])
        self.conv2 = nn.ModuleList([
            Block(
                in_channels=32,
                out_channels=64,
            )
        ])
        self.conv3 = nn.ModuleList([
            Block(
                in_channels=64,
                out_channels=128,
            )
        ])
        self.conv4 = nn.ModuleList([
            Block(
                in_channels=128,
                out_channels=256,
            )
        ])
        self.conv5 = nn.ModuleList([
            Block(
                in_channels=256,
                out_channels=512,
            )
        ])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=512*2*2, out_features=num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        for layer in self.conv1:
            x = layer(x)
        for layer in self.conv2:
            x = layer(x)
        for layer in self.conv3:
            x = layer(x)
        for layer in self.conv4:
            x = layer(x)
        for layer in self.conv5:
            x = layer(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def summary(self):
        summary(ResNet15_v1(), input_size=(2, 32, 32), batch_size=-1)
