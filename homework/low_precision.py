from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


def block_quantize_4bit(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize the input tensor to 4-bit precision along the last dimension.
    Always quantize group_size value together and store their absolute value first.
    To keep things simple, we require x to be a 1D tensor, and the size divisible by group_size.
    Return the quantized tensor and scaling factor.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    x_norm = (x + normalization) / (2 * normalization)
    x_quant_8 = (x_norm * 15).round().to(torch.int8)
    x_quant_4 = (x_quant_8[:, ::2] & 0xF) + ((x_quant_8[:, 1::2] & 0xF) << 4)
    return x_quant_4, normalization.to(torch.float16)


def block_dequantize_4bit(x_quant_4: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    """
    The reverse operation of block_quantize_4bit.
    x_quant_4: (num_groups, packed_per_group)  -> returns flat vector of length num_groups * group_size
    """
    assert x_quant_4.dim() == 2, f"x_quant_4 must be 2D, got {x_quant_4.dim()}"
    assert normalization.dim() == 2, f"normalization must be 2D, got {normalization.dim()}"

    normalization = normalization.to(torch.float32)
    x_quant_8 = x_quant_4.new_empty(x_quant_4.size(0), x_quant_4.shape[1] * 2)
    x_quant_8[:, ::2] = x_quant_4 & 0xF
    x_quant_8[:, 1::2] = (x_quant_4 >> 4) & 0xF
    x_norm = x_quant_8.to(torch.float32) / 15
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)


class Linear4Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 16) -> None:
        super().__init__()
        # Let's store all the required information to load the weights from a checkpoint
        self._shape = (out_features, in_features)
        self._group_size = group_size
        self.out_features = out_features
        self.in_features = in_features

        # number of groups per row
        groups_per_row = in_features // group_size
        # total number of groups across whole weight matrix
        total_groups = out_features * groups_per_row
        packed_per_group = group_size // 2  # two 4-bit per byte

        # registered buffers (2D): (total_groups, packed_per_group)
        self.register_buffer(
            "weight_q4",
            torch.zeros(total_groups, packed_per_group, dtype=torch.int8),
            persistent=False,
        )
        # normalization per group (total_groups, 1)
        self.register_buffer(
            "weight_norm",
            torch.zeros(total_groups, 1, dtype=torch.float16),
            persistent=False,
        )
        # Register a hook to load the weights from a checkpoint.
        self._register_load_state_dict_pre_hook(Linear4Bit._load_state_dict_pre_hook, with_module=True)
        # Add in an optional bias
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            # Load the original weights and remove them from the state_dict (mark them as loaded)
            weight = state_dict[f"{prefix}weight"]  # shape: (out_features, in_features)
            del state_dict[f"{prefix}weight"]

            q4_rows = []
            norm_rows = []
            for row in weight:
                qrow, nrow = block_quantize_4bit(row.detach().view(-1), self._group_size)
                # qrow: (groups_per_row, packed_per_group)
                # nrow: (groups_per_row, 1)
                q4_rows.append(qrow)
                norm_rows.append(nrow)

            # Stack into (out_features, groups_per_row, packed)
            q4_stacked = torch.stack(q4_rows)       # (out_features, groups_per_row, packed)
            norm_stacked = torch.stack(norm_rows)   # (out_features, groups_per_row, 1)

            # Flatten first two dims -> (total_groups, packed) to match registered buffers
            total_groups = q4_stacked.shape[0] * q4_stacked.shape[1]
            q4_flat = q4_stacked.view(total_groups, q4_stacked.shape[2]).to(torch.int8)
            norm_flat = norm_stacked.view(total_groups, 1).to(torch.float16)

            # copy into the registered buffers (do not reassign)
            self.weight_q4.copy_(q4_flat)
            self.weight_norm.copy_(norm_flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize the full weight matrix once (no_grad so we don't track ops on the buffers)
        with torch.no_grad():
            # self.weight_q4: (total_groups, packed)
            # self.weight_norm: (total_groups, 1)
            flat = block_dequantize_4bit(self.weight_q4, self.weight_norm)
            # flat length should be total_groups * group_size == out_features * in_features
            weight = flat.view(self.out_features, self.in_features)
        # linear should track gradients w.r.t. x (weight is treated as constant), so do not wrap the linear op in no_grad
        return torch.nn.functional.linear(x, weight, self.bias)


class BigNet4Bit(torch.nn.Module):
    """
    A BigNet where all weights are in 4bit precision. Use the Linear4Bit module for this.
    It is fine to keep all computation in float32.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear4Bit(channels, channels),
                torch.nn.ReLU(),
                Linear4Bit(channels, channels),
                torch.nn.ReLU(),
                Linear4Bit(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> BigNet4Bit:
    net = BigNet4Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net




# from pathlib import Path

# import torch

# from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


# def block_quantize_4bit(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Quantize the input tensor to 4-bit precision along the last dimension.
#     Always quantize group_size value together and store their absolute value first.
#     To keep things simple, we require x to be a 1D tensor, and the size divisible by group_size.
#     Return the quantized tensor and scaling factor.
#     """
#     assert x.dim() == 1
#     assert x.size(0) % group_size == 0

#     x = x.view(-1, group_size)
#     normalization = x.abs().max(dim=-1, keepdim=True).values
#     x_norm = (x + normalization) / (2 * normalization)
#     x_quant_8 = (x_norm * 15).round().to(torch.int8)
#     x_quant_4 = (x_quant_8[:, ::2] & 0xF) + ((x_quant_8[:, 1::2] & 0xF) << 4)
#     return x_quant_4, normalization.to(torch.float16)


# def block_dequantize_4bit(x_quant_4: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
#     """
#     The reverse operation of block_quantize_4bit.
#     """
#     assert x_quant_4.dim() == 2

#     normalization = normalization.to(torch.float32)
#     x_quant_8 = x_quant_4.new_empty(x_quant_4.size(0), x_quant_4.shape[1] * 2)
#     x_quant_8[:, ::2] = x_quant_4 & 0xF
#     x_quant_8[:, 1::2] = (x_quant_4 >> 4) & 0xF
#     x_norm = x_quant_8.to(torch.float32) / 15
#     x = (x_norm * 2 * normalization) - normalization
#     return x.view(-1)


# class Linear4Bit(torch.nn.Module):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 16) -> None:
#         super().__init__()
#         # Let's store all the required information to load the weights from a checkpoint
#         self._shape = (out_features, in_features)
#         self._group_size = group_size
#         self.out_features = out_features
#         self.in_features = in_features
#         # self.register_buffer is used to store the weights in the model, but not as parameters
#         # This makes sure weights are put on the correct device when calling `model.to(device)`.
#         # persistent=False makes sure the buffer is not saved or loaded. The bignet has a parameters
#         # called "weight" that we need to quantize when the model is loaded.
#         self.register_buffer(
#             "weight_q4",
#             torch.zeros(out_features * in_features // group_size, group_size // 2, dtype=torch.int8),
#             persistent=False,
#         )
#         self.register_buffer(
#             "weight_norm",
#             torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
#             persistent=False,
#         )
#         # Register a hook to load the weights from a checkpoint. This function reaches deep into
#         # PyTorch internals. It makes sure that Linear4Bit._load_state_dict_pre_hook is called
#         # every time the model is loaded from a checkpoint. We will quantize the weights in that function.
#         self._register_load_state_dict_pre_hook(Linear4Bit._load_state_dict_pre_hook, with_module=True)
#         # Add in an optional bias
#         self.bias = None
#         if bias:
#             self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

#     def _load_state_dict_pre_hook(
#         self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
#     ):
#         if f"{prefix}weight" in state_dict:
#             # Load the original weights and remove them from the state_dict (mark them as loaded)
#             weight = state_dict[f"{prefix}weight"]  # noqa: F841
#             del state_dict[f"{prefix}weight"]
#             q4, norm = [], []
#             for row in weight:
#                 qrow, nrow = block_quantize_4bit(row.detach().view(-1), self._group_size)
#                 q4.append(qrow)
#                 norm.append(nrow)
#             self.weight_q4 = torch.stack(q4).to(torch.uint8)
#             self.weight_norm = torch.stack(norm).to(torch.float16)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         with torch.no_grad():
#             weight = block_dequantize_4bit(self.weight_q4, self.weight_norm)
#             weight = weight.view(self.out_features, self.in_features)
#             return torch.nn.functional.linear(x, weight, self.bias)


# class BigNet4Bit(torch.nn.Module):
#     """
#     A BigNet where all weights are in 4bit precision. Use the Linear4Bit module for this.
#     It is fine to keep all computation in float32.
#     """

#     class Block(torch.nn.Module):
#         def __init__(self, channels):
#             super().__init__()
#             self.model = torch.nn.Sequential(
#             Linear4Bit(channels, channels),
#             torch.nn.ReLU(),
#             Linear4Bit(channels, channels),
#             torch.nn.ReLU(),
#             Linear4Bit(channels, channels)
#         )

#         def forward(self, x: torch.Tensor) -> torch.Tensor:
#             return self.model(x) + x

#     def __init__(self):
#         super().__init__()
#         self.model = torch.nn.Sequential(
#             self.Block(BIGNET_DIM),
#             LayerNorm(BIGNET_DIM),
#             self.Block(BIGNET_DIM),
#             LayerNorm(BIGNET_DIM),
#             self.Block(BIGNET_DIM),
#             LayerNorm(BIGNET_DIM),
#             self.Block(BIGNET_DIM),
#             LayerNorm(BIGNET_DIM),
#             self.Block(BIGNET_DIM),
#             LayerNorm(BIGNET_DIM),
#             self.Block(BIGNET_DIM),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.model(x)


# def load(path: Path | None) -> BigNet4Bit:
#     net = BigNet4Bit()
#     if path is not None:
#         net.load_state_dict(torch.load(path, weights_only=True))
#     return net

# #jainag