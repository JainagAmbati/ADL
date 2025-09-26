from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit


class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)
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


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            self.model = torch.nn.Sequential(
            QLoRALinear(channels, channels, lora_dim),
            torch.nn.ReLU(),
            QLoRALinear(channels, channels, lora_dim),
            torch.nn.ReLU(),
            QLoRALinear(channels, channels, lora_dim),
        )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 4, group_size: int = 16):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
