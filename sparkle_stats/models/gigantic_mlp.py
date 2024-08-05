from torch import nn

from sparkle_stats.sample_parameters import PARAMETER_COUNT


class GiganticMLP(nn.Module):
    """Large MLP model."""

    def __init__(self, output_classes=PARAMETER_COUNT, input_size=4000):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_classes),
        )

    def forward(self, raw):
        return self.network(raw)
