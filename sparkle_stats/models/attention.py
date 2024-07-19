import torch
from torch import nn

from sparkle_stats.sample_parameters import PARAMETER_COUNT


class Attention(nn.Module):
    def __init__(
        self,
        output_classes=PARAMETER_COUNT,
        input_size=4000,
    ):
        super().__init__()

        self.input_size = input_size

        self.attention1 = nn.MultiheadAttention(
            embed_dim=input_size, num_heads=8, dropout=0.1
        )
        self.attention2 = nn.MultiheadAttention(
            embed_dim=input_size, num_heads=8, dropout=0.1
        )
        self.attention3 = nn.MultiheadAttention(
            embed_dim=input_size, num_heads=8, dropout=0.1
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_classes),
        )

    def forward(self, raw):
        out, _ = self.attention1(raw, raw, raw)
        out, _ = self.attention2(out, out, out)
        out, _ = self.attention3(out, out, out)
        out = torch.sum(out, dim=1, keepdim=True)
        out = self.mlp(out)

        return out
