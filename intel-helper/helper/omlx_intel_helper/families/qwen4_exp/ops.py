# SPDX-License-Identifier: Apache-2.0
"""Checkpoint access and the small ops the Qwen3.8 Flash-Next blocks share (PyTorch).

Plain-op reference paths: the CUDA helper's fused kernels and its imitation of
Metal's staged 16-bit reductions are not ported, so 16-bit sums here round once
from a float32 accumulation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ... import checkpoint
from ...tensors import _unpack_rows
from ..qwen3_5.layers import rms_norm


class Linear:
    """A bias-free dense linear layer (one rounding from a float32 sum)."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight

    def parameters(self) -> list[torch.Tensor]:
        return [self.weight]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class Weights:
    """Checkpoint tensors under one name prefix, with their quantization."""

    def __init__(self, arrays: dict[str, torch.Tensor], config: dict[str, Any], root: str):
        self.arrays, self.config, self.root = arrays, config, root

    def at(self, path: str) -> Weights:
        return Weights(self.arrays, self.config, f"{self.root}{path}.")

    def __contains__(self, name: str) -> bool:
        return self.root + name in self.arrays

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.arrays[self.root + name]

    def quantization(self, name: str) -> dict[str, Any] | None:
        return checkpoint.quantization(self.config, self.root + name)

    def float32(self, name: str) -> torch.Tensor:
        """A linear weight in float32; a quantized one is not rounded."""
        weight = self[f"{name}.weight"]
        if f"{name}.scales" in self:
            weight = _unpack_rows(
                weight,
                self[f"{name}.scales"],
                self[f"{name}.biases"],
                self.quantization(name),
                torch.float32,
            )
        return weight.float()

    def linear(self, name: str) -> Linear:
        """A dense linear layer in the dtype the checkpoint computes in."""
        dtype = self[f"{name}.weight"].dtype
        if f"{name}.scales" in self:
            dtype = self[f"{name}.scales"].dtype
        return Linear(self.float32(name).to(dtype))


def unit_scale(weight: torch.Tensor) -> torch.Tensor:
    """Scale of a norm whose checkpoint weight is centered at zero."""
    return 1.0 + weight.float()


def norm(x: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS norm with a float32 scale, rounded once to the dtype of ``x``."""
    return rms_norm(x, scale, eps)


def stream_norm(x: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    """RMS norm of each residual stream; ``scale`` is [streams, hidden] float32."""
    streams, hidden = scale.shape
    y = x.reshape(*x.shape[:-1], streams, hidden).float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps) * scale
    return y.reshape(x.shape).to(x.dtype)


def row_sum(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=-1, keepdim=True)


def stacked_sum(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=-2)


def take(x: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """``x`` rows at ``index``; index_select, since advanced indexing crashes
    some torch XPU builds on narrow rows."""
    return torch.index_select(x, dim, index.long())


def depthwise_conv(x: torch.Tensor, taps: torch.Tensor, dilation: int) -> torch.Tensor:
    """Causal depthwise conv of [1, rows, C] with float32 ``taps`` [C, K].

    The taps add up in float32 in order and round once, like the host conv.
    """
    rows = x.shape[1] - (taps.shape[1] - 1) * dilation
    acc = x[:, :rows].float() * taps[:, 0]
    for i in range(1, taps.shape[1]):
        lo = i * dilation
        acc = acc + x[:, lo : lo + rows].float() * taps[:, i]
    return acc.to(x.dtype)
