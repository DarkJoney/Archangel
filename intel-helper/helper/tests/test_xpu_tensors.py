# SPDX-License-Identifier: Apache-2.0
"""tensors.py on the test device (Intel GPU, or CPU): MLX affine packing and the wire round trip."""

import numpy as np
import pytest
import torch


from omlx_intel_helper import tensors  # noqa: E402


def pack(codes: np.ndarray, bits: int) -> np.ndarray:
    """MLX layout: code j of a word sits at bits [j*bits, (j+1)*bits)."""
    per = 32 // bits
    words = np.zeros((codes.shape[0], codes.shape[1] // per), np.uint32)
    for j in range(per):
        words |= codes[:, j::per].astype(np.uint32) << np.uint32(j * bits)
    return words


@pytest.mark.parametrize("bits", [4, 8])
def test_unpack_matches_numpy_reference(bits):
    rng = np.random.default_rng(0)
    rows, cols, group = 96, 256, 64
    codes = rng.integers(0, 2**bits, (rows, cols))
    scales = (rng.random((rows, cols // group)) * 0.05).astype(np.float32)
    biases = (rng.random((rows, cols // group)) - 0.5).astype(np.float32)
    expected = (
        codes.reshape(rows, -1, group) * scales[..., None] + biases[..., None]
    ).reshape(rows, cols)
    weight = torch.from_numpy(pack(codes, bits).view(np.int32)).to(tensors.DEVICE)
    got = tensors.unpack(
        weight,
        torch.from_numpy(scales).to(tensors.DEVICE).bfloat16(),
        torch.from_numpy(biases).to(tensors.DEVICE).bfloat16(),
        {"bits": bits, "group_size": group},
        torch.bfloat16,
    )
    ref = torch.from_numpy(
        codes.reshape(rows, -1, group)
        * scales.astype(np.float32).__array__()[..., None]
    )  # only to check shape below
    assert got.shape == (rows, cols) and got.dtype == torch.bfloat16
    # Reference with the same bf16 scales/biases, rounded once.
    s16 = torch.from_numpy(scales).bfloat16().float().numpy()
    b16 = torch.from_numpy(biases).bfloat16().float().numpy()
    exact = (codes.reshape(rows, -1, group) * s16[..., None] + b16[..., None]).reshape(
        rows, cols
    )
    want = torch.from_numpy(exact.astype(np.float32)).bfloat16()
    assert torch.equal(got.cpu(), want)


def test_wire_round_trip():
    for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.int32):
        x = (torch.randn(5, 7) * 10).to(dtype).to(tensors.DEVICE)
        back = tensors.from_host(tensors.to_host(x))
        assert back.dtype == x.dtype and torch.equal(back.cpu(), x.cpu())
    name = tensors.to_host(torch.zeros(2, dtype=torch.bfloat16, device=tensors.DEVICE))[1]
    assert name == "bfloat16"
