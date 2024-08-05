# 1D ResNet implementation based on code from
# https://github.com/funkelab/funlib.learn.torch/blob/master/funlib/learn/torch/models/resnet2d.py

import torch
from torch import nn


class ResNet1D(nn.Module):
    """1D ResNet model."""

    def __init__(self, output_classes, input_channels=1, start_channels=12):
        """
        Args:
            output_classes: Number of output classes

            input_channels: Number of input channels

            start_channels: Number of channels in first convolutional layer
        """
        super().__init__()
        self.in_channels = start_channels
        self.conv = nn.Conv1d(
            input_channels,
            self.in_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
        )
        self.bn = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU()
        self.output_classes = output_classes

        current_channels = self.in_channels
        self.layer1 = self.make_layer(ResidualBlock, current_channels, 2, 2)
        current_channels *= 2
        self.layer2 = self.make_layer(ResidualBlock, current_channels, 2, 2)
        current_channels *= 2
        self.layer3 = self.make_layer(ResidualBlock, current_channels, 2, 2)
        current_channels *= 2
        self.layer4 = self.make_layer(ResidualBlock, current_channels, 2, 2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(current_channels, output_classes)
        self.relu = nn.ReLU()

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                    bias=True,
                ),
                nn.BatchNorm1d(out_channels),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = torch.hstack(
            (
                out[:, : self.output_classes // 2],
                self.relu(out[:, self.output_classes // 2 :]),
            )
        )
        return out


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # Biases are handled by BN layers
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=True,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = nn.ReLU()(out)
        return out
