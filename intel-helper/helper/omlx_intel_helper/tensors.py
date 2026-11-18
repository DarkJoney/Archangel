# SPDX-License-Identifier: Apache-2.0
"""PyTorch XPU helpers shared by the family stages (workers only).

Replaces the MLX layer of the CUDA helper: loading an MLX safetensors
checkpoint, moving arrays between the wire (numpy + MLX dtype name) and the
GPU, and the affine 4/8-bit unpack that MLX's ``dequantize`` does.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from . import checkpoint
from .wire import HostArray

# The XPU, unless OMLX_HELPER_DEVICE says otherwise ("cpu" runs the stages on
# the CPU: the tests use it where no Intel GPU is free; never a real helper).
DEVICE = torch.device(os.environ.get("OMLX_HELPER_DEVICE") or "xpu")
IS_XPU = DEVICE.type == "xpu"


def device_name() -> str:
    return torch.xpu.get_device_properties(0).name if IS_XPU else "CPU"


def synchronize() -> None:
    if IS_XPU:
        torch.xpu.synchronize()


def empty_cache() -> None:
    if IS_XPU:
        torch.xpu.empty_cache()

# MLX dtype name <-> torch dtype. bfloat16 crosses the wire as uint16 bits.
_TORCH = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}
# uint16 / uint32 have no full torch support: the same bits as signed types.
_BITS = {"uint16": torch.int16, "uint32": torch.int32, "uint64": torch.int64}

_limit = 0


def limit_memory(limit: int) -> None:
    """Remember the byte limit; the stage checks it with ``memory_bytes``.

    torch.xpu has no allocator cap, so a stage counts its own bytes against
    this before it runs a chunk (see ``supervisor.gpus.memory_limit``).
    """
    global _limit
    _limit = int(limit)


def memory_limit() -> int:
    return _limit


def allocated() -> int:
    return int(torch.xpu.memory_allocated()) if IS_XPU else 0


def peak_bytes() -> int:
    """Peak GPU bytes since the last call."""
    if not IS_XPU:
        return 0
    peak = int(torch.xpu.max_memory_allocated())
    torch.xpu.reset_peak_memory_stats()
    return peak


def check_backend() -> None:
    """Refuse a working-looking XPU whose bf16 matrix multiply is wrong.

    Without an XPU this returns: the fake family of the tests has no GPU, and a
    real family's stage fails with a clear error where it first needs the GPU.
    """
    if not IS_XPU or not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return
    torch.manual_seed(0)
    a = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(256, 256, device=DEVICE, dtype=torch.bfloat16)
    ref = (a.float().cpu() @ b.float().cpu())
    got = (a @ b).float().cpu()
    error = (got - ref).abs().max().item() / ref.abs().max().item()
    if not error < 0.05:
        raise RuntimeError(f"bf16 matmul on the XPU is off by {error:.3f}")


def dtype_name(array: torch.Tensor) -> str:
    return str(array.dtype).rsplit(".", 1)[-1]


def to_host(array: torch.Tensor) -> HostArray:
    """An array as contiguous numpy data and its MLX dtype name."""
    name = dtype_name(array)
    array = array.contiguous()
    if array.dtype == torch.bfloat16:
        # numpy has no bfloat16; the bits travel as uint16.
        data = array.view(torch.int16).cpu().numpy().view(np.uint16)
    else:
        data = array.cpu().numpy()
    return np.ascontiguousarray(data), name


def from_host(value: HostArray) -> torch.Tensor:
    data, name = value
    if name == "bfloat16":
        t = torch.from_numpy(np.ascontiguousarray(data).view(np.int16)).view(
            torch.bfloat16
        )
    elif name in _BITS:
        view = {"uint16": np.int16, "uint32": np.int32, "uint64": np.int64}[name]
        t = torch.from_numpy(np.ascontiguousarray(data).view(view))
    else:
        t = torch.from_numpy(np.ascontiguousarray(data))
    return t.to(DEVICE)


# Set by the worker where a GPU reads host memory slowly; kept for the common
# stage code. XPU weights always load to the device here.
weights_on_gpu = False


def load(model_dir: str | Path, names: list[str]) -> dict[str, torch.Tensor]:
    """The named tensors on the XPU. Each shard file is opened once."""
    root = Path(model_dir)
    index = checkpoint.tensors(root)
    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(index[name].file, []).append(name)
    out: dict[str, torch.Tensor] = {}
    for file, wanted in by_file.items():
        with safe_open(str(root / file), framework="pt", device="cpu") as f:
            for name in wanted:
                tensor = f.get_tensor(name)
                # MLX stores quantized weights as uint32; keep the bits.
                if tensor.dtype == torch.uint32:
                    tensor = tensor.view(torch.int32)
                out[name] = tensor.to(DEVICE)
    return out


def unpack(
    weight: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor | None,
    quant: dict[str, Any],
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """The dense weight of an affine quantized tensor, rounded once.

    Like MLX's: rebuilt in float32 and rounded to the target type afterwards.
    ``weight`` is the packed int32 [..., rows, cols * bits / 32]; ``scales`` and
    ``biases`` are [..., rows, cols / group_size].
    """
    mode = quant.get("mode", "affine")
    if mode != "affine":
        raise NotImplementedError(f"quantization mode {mode!r} on the XPU")
    bits, group = int(quant["bits"]), int(quant["group_size"])
    target = dtype or scales.dtype
    *lead, packed = weight.shape  # leading dims: rows, or experts and rows
    per_word = 32 // bits
    shifts = torch.arange(0, 32, bits, device=weight.device, dtype=torch.int32)
    mask = (1 << bits) - 1
    codes = (weight.unsqueeze(-1) >> shifts) & mask  # [..., packed, per_word]
    codes = codes.reshape(*lead, packed * per_word).to(torch.float32)
    grouped = codes.reshape(*lead, -1, group)
    dense = grouped * scales.float().unsqueeze(-1)
    if biases is not None:
        dense = dense + biases.float().unsqueeze(-1)
    return dense.reshape(*lead, -1).to(target)


class Linear:
    """A bias-free linear layer from a dense or a quantized weight.

    ``dense=True`` unpacks a quantized weight at load: the dense product is
    the fast path (no unpack per chunk) and costs the unpacked bytes of GPU
    memory.
    """

    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        prefix: str,
        quant: dict[str, Any] | None,
        *,
        dense: bool = True,
        dtype: torch.dtype | None = None,
    ):
        weight = weights[f"{prefix}.weight"]
        self.quant = None
        if f"{prefix}.scales" in weights:
            scales = weights[f"{prefix}.scales"]
            biases = weights.get(f"{prefix}.biases")
            # Unpack in chunks of rows so float32 temporaries stay small.
            weight = _unpack_rows(weight, scales, biases, quant, dtype)
            if not dense:
                raise NotImplementedError("fused quantized matmul on the XPU")
        elif dtype is not None:
            weight = weight.to(dtype)
        self.weight = weight

    def parameters(self) -> list[torch.Tensor]:
        return [self.weight]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


def _unpack_rows(weight, scales, biases, quant, dtype, rows: int = 2048):
    parts = []
    for i in range(0, weight.shape[0], rows):
        parts.append(
            unpack(
                weight[i : i + rows],
                scales[i : i + rows],
                None if biases is None else biases[i : i + rows],
                quant,
                dtype,
            )
        )
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0") == "1"
