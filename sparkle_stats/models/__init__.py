from .attention import Attention
from .conv_sfc import ConvSFC
from .gigantic_mlp import GiganticMLP
from .resnet1d import ResNet1D
from .simple_fully_connected import SimpleFullyConnected
from .vgg1d import Vgg1D
from .vgg_simple import SimpleVgg

__all__ = [
    "Attention",
    "ConvSFC",
    "GiganticMLP",
    "ResNet1D",
    "SimpleFullyConnected",
    "SimpleVgg",
    "Vgg1D",
]
