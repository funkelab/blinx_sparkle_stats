# %%
import torch
from torch import nn


# %%
class Attention(nn.Module):
    def __init__(
        self,
        input_channels=1,
        embed_dim=4,
        input_size=4000,
        output_classes=7,
        num_heads=2,
        dropout=0.1,
    ):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"
        self.input_size = input_size

        self.embedding = nn.Linear(input_channels, embed_dim)

        self.attention1 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
        )
        self.attention2 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
        )
        self.attention3 = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_classes),
        )

    def forward(self, raw):
        # raw is of shape (batch_size, channel, input_size)
        # convert to (batch_size, input_size, channel)
        out = raw.permute(0, 2, 1)
        out = self.embedding(out)
        out, matrix = self.attention1(out, out, out)
        # print(matrix.shape)
        # mlp wants (batch_size, channel, input_size)
        out = out.permute(0, 2, 1)
        out = self.mlp1(out)

        out = out.permute(0, 2, 1)
        out, _ = self.attention2(out, out, out)
        out = out.permute(0, 2, 1)
        out = self.mlp2(out)

        out = out.permute(0, 2, 1)
        out, _ = self.attention3(out, out, out)
        out = out.permute(0, 2, 1)
        out = self.mlp3(out)

        out = torch.sum(out, dim=1, keepdim=True)

        return out
