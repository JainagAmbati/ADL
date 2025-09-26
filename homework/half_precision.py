from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm

class HalfLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.to(dtype=torch.float16)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float16)
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        return out

class Block(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.model = torch.nn.Sequential(
            HalfLinear(channels, channels),
            torch.nn.ReLU(),
            HalfLinear(channels, channels),
            torch.nn.ReLU(),
            HalfLinear(channels, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) + x

class HalfBigNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def load(path: Path = None) -> HalfBigNet:
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net