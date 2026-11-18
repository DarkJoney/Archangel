# SPDX-License-Identifier: Apache-2.0
"""The qwen3_5 stage on a real Intel GPU with a tiny random MLX-layout model.

No reference model is needed for these: they check that the stage is
self-consistent the way the split has to be (chunking, cutting between two
stages and restoring a prefix must not change the numbers).
"""

import json

import numpy as np
import pytest
import torch

from safetensors.torch import save_file  # noqa: E402

from omlx_intel_helper.families.qwen3_5.stage import load_stage  # noqa: E402
from omlx_intel_helper.tensors import from_host, to_host  # noqa: E402

HIDDEN, INTER, LAYERS, VOCAB = 128, 256, 4, 512
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 32
LK, LV, DK, DV, KERNEL = 2, 4, 32, 32, 4
GROUP, BITS = 64, 4


def pack(codes: np.ndarray) -> torch.Tensor:
    per = 32 // BITS
    words = np.zeros((codes.shape[0], codes.shape[1] // per), np.uint32)
    for j in range(per):
        words |= codes[:, j::per].astype(np.uint32) << np.uint32(j * BITS)
    return torch.from_numpy(words.view(np.int32)).view(torch.uint32)


def quantized(rng, out, inp, scale=0.02, dtype=torch.bfloat16):
    codes = rng.integers(0, 2**BITS, (out, inp))
    s = (rng.random((out, inp // GROUP)) * scale + 0.005).astype(np.float32)
    b = (-s * 2**BITS / 2).astype(np.float32)
    return {
        "weight": pack(codes),
        "scales": torch.from_numpy(s).to(dtype),
        "biases": torch.from_numpy(b).to(dtype),
    }


def make_model(path, dtype=torch.bfloat16):
    rng = np.random.default_rng(0)
    text = {
        "hidden_size": HIDDEN, "intermediate_size": INTER, "vocab_size": VOCAB,
        "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS, "head_dim": HEAD_DIM,
        "linear_num_key_heads": LK, "linear_num_value_heads": LV,
        "linear_key_head_dim": DK, "linear_value_head_dim": DV,
        "linear_conv_kernel_dim": KERNEL, "rms_norm_eps": 1e-6,
        "full_attention_interval": 4,
        "rope_parameters": {"rope_theta": 1_000_000.0, "partial_rotary_factor": 0.25},
    }
    tensors = {}
    root = "language_model.model."

    def add(prefix, out, inp):
        for part, value in quantized(rng, out, inp, dtype=dtype).items():
            tensors[f"{prefix}.{part}"] = value

    def vec(n, base=1.0, spread=0.1):
        values = base + spread * rng.standard_normal(n)
        return torch.from_numpy(values.astype(np.float32)).to(dtype)

    add(f"{root}embed_tokens", VOCAB, HIDDEN)
    for i in range(LAYERS):
        p = f"{root}layers.{i}."
        tensors[p + "input_layernorm.weight"] = vec(HIDDEN)
        tensors[p + "post_attention_layernorm.weight"] = vec(HIDDEN)
        for name, (o, n) in {
            "gate_proj": (INTER, HIDDEN),
            "up_proj": (INTER, HIDDEN),
            "down_proj": (HIDDEN, INTER),
        }.items():
            add(p + "mlp." + name, o, n)
        if (i + 1) % 4:
            q = p + "linear_attn."
            conv = 2 * LK * DK + LV * DV
            add(q + "in_proj_qkv", conv, HIDDEN)
            add(q + "in_proj_z", LV * DV, HIDDEN)
            add(q + "in_proj_b", LV, HIDDEN)
            add(q + "in_proj_a", LV, HIDDEN)
            add(q + "out_proj", HIDDEN, LV * DV)
            taps = rng.standard_normal((conv, KERNEL, 1)) * 0.3
            tensors[q + "conv1d.weight"] = torch.from_numpy(taps.astype(np.float32)).to(dtype)
            tensors[q + "A_log"] = torch.from_numpy(
                (rng.standard_normal(LV) * 0.3).astype(np.float32)
            )
            tensors[q + "dt_bias"] = vec(LV, 0.5, 0.1)
            tensors[q + "norm.weight"] = vec(DV)
        else:
            q = p + "self_attn."
            add(q + "q_proj", HEADS * HEAD_DIM * 2, HIDDEN)
            add(q + "k_proj", KV_HEADS * HEAD_DIM, HIDDEN)
            add(q + "v_proj", KV_HEADS * HEAD_DIM, HIDDEN)
            add(q + "o_proj", HIDDEN, HEADS * HEAD_DIM)
            tensors[q + "q_norm.weight"] = vec(HEAD_DIM)
            tensors[q + "k_norm.weight"] = vec(HEAD_DIM)
    save_file(tensors, str(path / "model.safetensors"))
    config = {"text_config": text, "quantization": {"group_size": GROUP, "bits": BITS}}
    (path / "config.json").write_text(json.dumps(config))


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny")
    make_model(path)
    return path


TOKENS = np.random.default_rng(1).integers(3, VOCAB, 200).astype(np.int32)


def rope(stage):
    return {"rope": to_host(stage.frequencies)}


def run_chunks(stage, sizes, job=None):
    """Feed the prompt in chunks; returns h rows and the concatenated state."""
    job = job or stage.new_job(len(stage.layers))
    start, hs, kv = 0, [], {}
    for size in sizes:
        out, state = stage.run(
            job, start=start, tokens=TOKENS[start : start + size], boundary=None,
            inputs=rope(stage), following=0, capture=None,
        )
        hs.append(from_host(out["h"]).float())
        for name, array in state.items():
            value = from_host(array)
            if name.endswith((".k", ".v")) and name in kv:
                value = torch.cat([kv[name], value], dim=2)
            kv[name] = value  # conv window / recurrent state: the latest
        start += size
    return torch.cat(hs, dim=1), kv, job


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_chunking_does_not_change_the_result(model):
    stage = load_stage(model, 0, LAYERS)
    whole, kv_whole, _ = run_chunks(stage, [200])
    split, kv_split, _ = run_chunks(stage, [64, 64, 72])
    assert torch.isfinite(whole).all()
    assert rel(split, whole) < 0.02
    for name in kv_whole:
        assert rel(kv_split[name], kv_whole[name]) < 0.05, name


def test_two_stages_equal_one(model):
    one = load_stage(model, 0, LAYERS)
    first, second = load_stage(model, 0, 2), load_stage(model, 2, LAYERS)
    want, _, _ = run_chunks(one, [100, 100])
    job_a, job_b = first.new_job(2), second.new_job(2)
    start, hs = 0, []
    for size in (100, 100):
        out, _ = first.run(
            job_a, start=start, tokens=TOKENS[start : start + size], boundary=None,
            inputs=rope(first), following=0, capture=None,
        )
        out2, _ = second.run(
            job_b, start=start, tokens=TOKENS[start : start + size], boundary=out,
            inputs={}, following=0, capture=None,
        )
        hs.append(from_host(out2["h"]).float())
        start += size
    assert rel(torch.cat(hs, dim=1), want) < 1e-2


def test_a_restored_prefix_continues_the_job(model):
    stage = load_stage(model, 0, LAYERS)
    want, _, _ = run_chunks(stage, [128, 72])
    job = stage.new_job(LAYERS)
    _, state = stage.run(
        job, start=0, tokens=TOKENS[:128], boundary=None, inputs=rope(stage),
        following=0, capture=None,
    )
    # What the host restores from its prefix cache: the whole state so far.
    fresh = stage.new_job(LAYERS)
    stage.load_prefix(fresh, state)
    out2, _ = stage.run(
        fresh, start=128, tokens=TOKENS[128:200], boundary=None, inputs=rope(stage),
        following=0, capture=None,
    )
    assert rel(from_host(out2["h"]).float(), want[:, 128:]) < 1e-2


def test_dropping_and_adding_a_layer(model):
    stage = load_stage(model, 0, LAYERS)
    want, _, _ = run_chunks(stage, [200])
    stage.drop_layer()
    assert stage.last == LAYERS - 1 and len(stage.layers) == LAYERS - 1
    stage.add_layer()
    again, _, _ = run_chunks(stage, [200])
    assert torch.equal(again, want)


def test_fused_attention_agrees_with_the_float32_sheet(model):
    stage = load_stage(model, 0, LAYERS)
    want, _, _ = run_chunks(stage, [200])
    for layer in stage.layers:
        if not layer.linear:
            layer.mixer.fused = True
    got, _, _ = run_chunks(stage, [200])
    assert rel(got, want) < 2e-2
