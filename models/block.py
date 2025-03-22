import torch.nn as nn

class Block(nn.Module):
    def __init__(
        self, in_channels:int, out_channels:int, residual=True
    ):
        super().__init__()
        self.residual = residual

        if(in_channels*2!=out_channels):
            raise ValueError("Number of out-channels in ResNet-15 are exactly double of in-channels")

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=2,
            padding=0,
        )
        self.batchnorm1 = nn.BatchNorm2d(
            num_features=in_channels,
            track_running_stats=True,
        )
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.batchnorm2 = nn.BatchNorm2d(
            num_features=in_channels,
            track_running_stats=True,
        )

        self.conv3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.batchnorm3 = nn.BatchNorm2d(
            num_features=out_channels,
            track_running_stats=True,
        )

        self.residual_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2,
            padding=0,
        )

        self.residual_bn = nn.BatchNorm2d(
            num_features=out_channels
        )

    def forward(self, x):
        z = self.conv1(x)
        z = self.batchnorm1(z)
        z = self.relu(z)
        z = self.conv2(z)
        z = self.batchnorm2(z)
        z = self.relu(z)
        z = self.conv3(z)
        z = self.batchnorm3(z)
        if self.residual:
            x = self.residual_conv(x)
            x = self.residual_bn(x)
            z = z + x
        x = self.relu(z)
        return x
