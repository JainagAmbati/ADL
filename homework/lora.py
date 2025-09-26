from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm
from .half_precision import HalfLinear


class LoRALinear(HalfLinear):
    def __init__(self, in_features, out_features, lora_dim, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        # LoRA adapters, float32 for training updates
        self.lora_a = torch.nn.Parameter(torch.randn(lora_dim, in_features) * 0.01)
        self.lora_b = torch.nn.Parameter(torch.randn(out_features, lora_dim) * 0.01)
        self.lora_scale = 0.8
        for param in self.parameters():
            param.requires_grad = False
        self.lora_a.requires_grad = True
        self.lora_b.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = super().forward(x)
        # LoRA output computed in float32 precision, then cast back
        lora_out = torch.nn.functional.linear(
            torch.nn.functional.linear(x.float(), self.lora_a), self.lora_b
        ) * self.lora_scale
        return base_out + lora_out.to(base_out.dtype)


class Block(torch.nn.Module):
    def __init__(self, channels, lora_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            LoRALinear(channels, channels, lora_dim),
            torch.nn.ReLU(),
            LoRALinear(channels, channels, lora_dim),
            torch.nn.ReLU(),
            LoRALinear(channels, channels, lora_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) + x


class LoraBigNet(torch.nn.Module):
    def __init__(self, lora_dim=8):
        super().__init__()
        self.model = torch.nn.Sequential(
            Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            Block(BIGNET_DIM, lora_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path = None, lora_dim=2) -> LoraBigNet:
    net = LoraBigNet(lora_dim)
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
