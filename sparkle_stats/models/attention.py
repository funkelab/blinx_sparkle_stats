import torch
from torch import nn

from sparkle_stats.sample_parameters import PARAMETER_COUNT


class Attention(nn.Module):
    def __init__(
        self,
        output_classes=PARAMETER_COUNT,
        input_size=4000,
        embed_dim=1,
        num_heads=1,
        dropout=0.1,
    ):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"
        self.input_size = input_size

        self.attention1 = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
        )
        self.attention2 = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
        )
        self.attention3 = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_classes),
        )

    def forward(self, raw):
        # raw is of shape (batch_size, channel, input_size)
        # attention wants (batch_size, input_size, channel)
        raw = raw.permute(0, 2, 1)
        out, _ = self.attention1(raw, raw, raw)
        # linear wants (batch_size, channel, input_size)
        out = out.permute(0, 2, 1)
        out = self.mlp1(out)

        out = out.permute(0, 2, 1)
        out, _ = self.attention2(out, out, out)
        out = out.permute(0, 2, 1)
        out = torch.sum(out, dim=1, keepdim=True)
        out = self.mlp2(out)

        out = out.permute(0, 2, 1)
        out, _ = self.attention3(out, out, out)
        out = out.permute(0, 2, 1)
        out = torch.sum(out, dim=1, keepdim=True)
        out = self.mlp3(out)

        return out
