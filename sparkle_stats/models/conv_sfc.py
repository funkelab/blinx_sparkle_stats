import torch
from torch import nn

from sparkle_stats.sample_parameters import PARAMETER_COUNT


class ConvSFC(nn.Module):
    def __init__(self, output_classes=PARAMETER_COUNT, input_size=4000):
        super().__init__()

        self.input_size = input_size

        self.conv_network = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(2, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc_network = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_classes),
        )

    def forward(self, raw):
        out = self.conv_network(raw)
        out = torch.sum(out, dim=1, keepdim=True)
        out = self.fc_network(out)

        return out
