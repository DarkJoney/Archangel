# SPDX-License-Identifier: Apache-2.0
"""Flash-Next with 4-bit affine weights equals its dense twin.

The Mac fixture model is dense. This test quantizes its linears, its expert
banks and its embedding (MLX layout, group size 16) and builds the dense twin
of exactly the dequantized values, then runs both through the stage with the
fixture's inputs: the packed paths (``Weights``, ``Bank``, the embedding
unpack) must give what the dense ones give.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from omlx_intel_helper.families.qwen4_exp.stage import load_stage
from omlx_intel_helper.tensors import from_host

FIXTURE = Path(__file__).parent / "fixtures/prefill_contract/qwen4_exp"
BITS, GROUP = 4, 16


def pack(codes: np.ndarray) -> torch.Tensor:
    """MLX layout: code j of a word sits at bits [j*BITS, (j+1)*BITS)."""
    per = 32 // BITS
    flat = codes.reshape(-1, codes.shape[-1])
    words = np.zeros((flat.shape[0], flat.shape[1] // per), np.uint32)
    for j in range(per):
        words |= flat[:, j::per].astype(np.uint32) << np.uint32(j * BITS)
    out = words.reshape(*codes.shape[:-1], codes.shape[-1] // per)
    return torch.from_numpy(out.view(np.int32)).view(torch.uint32)


def quantize(weight: torch.Tensor):
    """(packed, scales, biases, dense twin) of a float32 [..., in] weight."""
    w = weight.numpy().astype(np.float64)
    groups = w.reshape(*w.shape[:-1], -1, GROUP)
    low, high = groups.min(-1, keepdims=True), groups.max(-1, keepdims=True)
    scale = np.maximum((high - low) / (2**BITS - 1), 1e-8)
    codes = np.clip(np.round((groups - low) / scale), 0, 2**BITS - 1)
    scale32, bias32 = scale.astype(np.float32), low.astype(np.float32)
    twin = (codes.astype(np.float32) * scale32 + bias32).reshape(w.shape)
    return (
        pack(codes.reshape(w.shape).astype(np.int64)),
        torch.from_numpy(scale32[..., 0]),
        torch.from_numpy(bias32[..., 0]),
        torch.from_numpy(twin),
    )


def quantizable(name: str, tensor: torch.Tensor) -> bool:
    skip = ("norm", "conv1d", "A_log", "dt_bias", "ple_embedding")
    return (
        name.endswith(".weight")
        and tensor.ndim in (2, 3)
        and tensor.dtype == torch.float32
        and tensor.shape[-1] % GROUP == 0
        and tensor.shape[-1] > 1
        and not any(part in name for part in skip)
    )


@pytest.fixture(scope="module")
def twins(tmp_path_factory):
    root = tmp_path_factory.mktemp("q4")
    dense_dir, packed_dir = root / "dense", root / "packed"
    for d in (dense_dir, packed_dir):
        d.mkdir()
    tensors = load_file(str(FIXTURE / "model" / "model.safetensors"))
    dense, packed, count = dict(tensors), dict(tensors), 0
    for name, tensor in tensors.items():
        if not quantizable(name, tensor):
            continue
        weight, scales, biases, twin = quantize(tensor)
        base = name.removesuffix(".weight")
        del packed[name]
        packed[name], packed[base + ".scales"], packed[base + ".biases"] = weight, scales, biases
        dense[name] = twin
        count += 1
    assert count > 10  # linears, expert banks and the embedding really changed
    save_file(dense, str(dense_dir / "model.safetensors"))
    save_file(packed, str(packed_dir / "model.safetensors"))
    config = json.loads((FIXTURE / "model" / "config.json").read_text())
    shutil.copy(FIXTURE / "model" / "config.json", dense_dir / "config.json")
    config["quantization"] = {"group_size": GROUP, "bits": BITS}
    (packed_dir / "config.json").write_text(json.dumps(config))
    return dense_dir, packed_dir


def _run(model: Path):
    meta = json.loads((FIXTURE / "meta.json").read_text())
    data = np.load(FIXTURE / "case.npz")
    arrays = {n: (data[n], d) for n, d, _, _ in meta["manifest"]}
    group = lambda tag: {n[len(tag):]: v for n, v in arrays.items() if n.startswith(tag)}  # noqa: E731
    k, sizes, skip = meta["layers"], meta["sizes"], meta["prefix_chunks"]
    stage = load_stage(model, 0, k)
    job = stage.new_job(k)
    stage.load_prefix(job, group("p:"))
    tokens = np.asarray(meta["ids"], dtype=np.int32)[sum(sizes[:skip]) :]
    start, hs = 0, []
    for index, size in enumerate(sizes[skip:]):
        out, _ = stage.run(
            job, start=start, tokens=tokens[start : start + size], boundary=None,
            inputs=group(f"in{index}:"), following=0, capture=None,
        )
        hs.append(from_host(out["h"]).float().cpu().numpy())
        start += size
    return np.concatenate(hs, axis=1)


def test_packed_weights_give_the_dense_twins_result(twins):
    dense_dir, packed_dir = twins
    want, got = _run(dense_dir), _run(packed_dir)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
