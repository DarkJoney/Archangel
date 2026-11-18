# SPDX-License-Identifier: Apache-2.0
"""Prefill forward of the Qwen3.5 blocks, batch one and text only (PyTorch).

Each block follows the path the oMLX host takes for a prefill chunk, op by op
and in the same dtypes, so 16-bit activations round as they do there. The
blocks are ports of the mlx-vlm classes (MIT, Copyright (c) 2025 Prince
Canuma); attention keeps scores, softmax and value sums in float32 by default.
"""

from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F

from ... import checkpoint
from ...tensors import Linear
from . import gdn

# Bytes of the float32 score sheet one attention step holds.
_SHEET_BYTES = 1 << 30
# Use torch's fused attention (16-bit scores) instead of the float32 sheet.
_FUSED = os.environ.get("OMLX_HELPER_SDPA", "0") == "1"


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Float32 norm, rounded once to the input type."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


class Weights:
    """Checkpoint tensors under one name prefix, with their quantization."""

    def __init__(self, arrays: dict[str, torch.Tensor], config: dict[str, Any], root: str):
        self.arrays, self.config, self.root = arrays, config, root

    def at(self, path: str) -> Weights:
        return Weights(self.arrays, self.config, f"{self.root}{path}.")

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.arrays[self.root + name]

    def linear(self, name: str) -> Linear:
        """A dense linear layer in the dtype the checkpoint computes in."""
        path = self.root + name
        return Linear(self.arrays, path, checkpoint.quantization(self.config, path))


def causal_conv(x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """Depthwise conv of [1, rows + K - 1, C] with float32 ``taps`` [C, K].

    The taps add up in float32 in order and round once, like the host conv.
    """
    rows = x.shape[1] - taps.shape[1] + 1
    acc = x[:, :rows].float() * taps[:, 0]
    for i in range(1, taps.shape[1]):
        acc = acc + x[:, i : i + rows].float() * taps[:, i]
    return acc.to(x.dtype)


class GatedDeltaNet:
    """Linear attention: a causal conv and the gated delta recurrence."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.key_heads = text["linear_num_key_heads"]
        self.value_heads = text["linear_num_value_heads"]
        self.key_dim = text["linear_key_head_dim"]
        self.value_dim = text["linear_value_head_dim"]
        self.eps = text["rms_norm_eps"]
        self.qkv = w.linear("in_proj_qkv")
        self.z = w.linear("in_proj_z")
        self.b = w.linear("in_proj_b")
        self.a = w.linear("in_proj_a")
        self.out = w.linear("out_proj")
        self.taps = w["conv1d.weight"][..., 0].float()
        self.decay = torch.exp(w["A_log"].float())
        self.dt_bias = w["dt_bias"]
        self.norm_weight = w["norm.weight"]

    def parameters(self) -> list[torch.Tensor]:
        linears = (self.qkv, self.z, self.b, self.a, self.out)
        own = [self.taps, self.decay, self.dt_bias, self.norm_weight]
        return own + [a for layer in linears for a in layer.parameters()]

    def prework(self, qkv: torch.Tensor, window: torch.Tensor | None):
        """(q, k, v, next conv window) from the projected rows [1, rows, C]."""
        rows = qkv.shape[1]
        if window is None:
            width = self.taps.shape[1] - 1
            window = torch.zeros(1, width, qkv.shape[-1], dtype=qkv.dtype, device=qkv.device)
        conv_in = torch.cat([window, qkv], dim=1)
        conv_out = F.silu(causal_conv(conv_in, self.taps))
        keys = self.key_heads * self.key_dim
        q = conv_out[..., :keys].reshape(1, rows, self.key_heads, self.key_dim)
        k = conv_out[..., keys : 2 * keys].reshape(q.shape)
        v = conv_out[..., 2 * keys :]
        v = v.reshape(1, rows, self.value_heads, self.value_dim)
        # L2 norm as an RMS norm; the query also takes the 1/sqrt(dim) of
        # the readout. Both scales multiply the rounded norm.
        scale = self.key_dim**-0.5
        eps = 1e-6 * scale**2
        q = (scale**2) * rms_norm(q, None, eps)
        k = scale * rms_norm(k, None, eps)
        return q, k, v, conv_in[:, rows:]

    def gates(self, a: torch.Tensor, b: torch.Tensor):
        """(decay in float32, update rate) of each step and value head."""
        return torch.exp(-self.decay * F.softplus(a + self.dt_bias).float()), torch.sigmoid(b)

    def finish(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """The recurrence output normalized and gated, [1, rows, value width]."""
        y = rms_norm(y, self.norm_weight, self.eps)
        gate = F.silu(z.reshape(y.shape).float())
        gated = (gate * y.float()).to(y.dtype)
        return gated.reshape(1, y.shape[1], -1)

    def __call__(self, x: torch.Tensor, cache: list) -> torch.Tensor:
        q, k, v, cache[0] = self.prework(self.qkv(x), cache[0])
        gamma, beta = self.gates(self.a(x), self.b(x))
        y, cache[1] = gdn.update(q, k, v, gamma, beta, cache[1])
        return self.out(self.finish(y, self.z(x)))


class KVCache:
    """Rotated keys and values of one attention layer for one prompt."""

    def __init__(self) -> None:
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.offset = 0
        # (keys, values) of the last chunk.
        self.appended: tuple[torch.Tensor, ...] = ()

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Add rows [1, kv heads, n, dim]."""
        self.appended = (keys, values)
        if self.keys is not None:
            keys = torch.cat([self.keys, keys], dim=2)
            values = torch.cat([self.values, values], dim=2)
        self.keys, self.values = keys, values
        self.offset = keys.shape[2]

    def arrays(self) -> list[torch.Tensor]:
        return [a for a in (self.keys, self.values) if a is not None]


def rotary_frequencies(text: dict[str, Any], device=None) -> torch.Tensor:
    """Inverse frequencies of the rotary dims as this backend rounds them.

    Only a warm-up chunk uses them: a prompt's chunks carry the frequencies
    of the host, whose last bits differ.
    """
    rope = text.get("rope_parameters") or {}
    heads = text["num_attention_heads"]
    head_dim = text.get("head_dim") or text["hidden_size"] // heads
    rotary = int(head_dim * rope.get("partial_rotary_factor", 0.25))
    exponent = torch.arange(0, rotary, 2, dtype=torch.float32, device=device) / rotary
    return 1.0 / (float(rope.get("rope_theta", 10_000_000)) ** exponent)


def rotate(x: torch.Tensor, positions: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Rotate the leading dims of [1, heads, n, dim] by text positions [n].

    Angles, sines and the rotation are float32 and the result rounds once.
    """
    half = frequencies.shape[0]
    angle = positions.float()[:, None] * frequencies
    cos, sin = torch.cos(angle), torch.sin(angle)
    a = x[..., :half].float()
    b = x[..., half : 2 * half].float()
    parts = [(a * cos - b * sin).to(x.dtype), (b * cos + a * sin).to(x.dtype)]
    if x.shape[-1] > 2 * half:
        parts.append(x[..., 2 * half :])
    return torch.cat(parts, dim=-1)


class Attention:
    """Gated causal attention over every cached token."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.heads = text["num_attention_heads"]
        self.kv_heads = text["num_key_value_heads"]
        self.head_dim = text.get("head_dim") or text["hidden_size"] // self.heads
        self.eps = text["rms_norm_eps"]
        self.q_proj = w.linear("q_proj")
        self.k_proj = w.linear("k_proj")
        self.v_proj = w.linear("v_proj")
        self.o_proj = w.linear("o_proj")
        self.q_norm = w["q_norm.weight"]
        self.k_norm = w["k_norm.weight"]
        self.fused = _FUSED

    def parameters(self) -> list[torch.Tensor]:
        linears = (self.q_proj, self.k_proj, self.v_proj, self.o_proj)
        own = [self.q_norm, self.k_norm]
        return own + [a for layer in linears for a in layer.parameters()]

    def attend(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, step: int = 0
    ) -> torch.Tensor:
        """Causal attention of the last rows of the cache over all of it.

        ``queries`` [1, heads, rows, dim] are the newest ``rows`` tokens of
        ``keys`` / ``values`` [1, kv heads, total, dim]. Returns
        [1, rows, heads, dim]. Scores, softmax and value sums are float32 and
        the output rounds once; ``step`` query rows share a score sheet, and 0
        sizes it to ``_SHEET_BYTES``.
        """
        rows, total = queries.shape[2], keys.shape[2]
        past = total - rows
        kv, dim = self.kv_heads, self.head_dim
        groups = self.heads // kv
        scale = 1.0 / math.sqrt(dim)
        if self.fused:
            k = keys.repeat_interleave(groups, dim=1)
            v = values.repeat_interleave(groups, dim=1)
            visible = (
                torch.arange(rows, device=queries.device)[:, None] + past
                >= torch.arange(total, device=queries.device)[None]
            )
            out = F.scaled_dot_product_attention(
                queries, k, v, attn_mask=visible, scale=scale
            )
            return out.transpose(1, 2)
        grouped = queries.reshape(1, kv, groups, rows, dim)
        # Transposed once: every step multiplies by the same keys.
        kt = keys.float().transpose(-1, -2).contiguous()
        vf = values.float()
        step = step or max(1, _SHEET_BYTES // (4 * self.heads * total))
        outputs = []
        cols = torch.arange(total, device=queries.device)
        for start in range(0, rows, step):
            stop = min(start + step, rows)
            n = stop - start
            q = grouped[..., start:stop, :].float() * scale
            scores = q.reshape(1, kv, groups * n, dim) @ kt
            visible = (past + torch.arange(start, stop, device=queries.device))[:, None] >= cols
            scores = scores.reshape(1, kv, groups, n, total)
            weights = torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)
            out = weights.reshape(1, kv, groups * n, total) @ vf
            outputs.append(out.to(queries.dtype).reshape(1, kv, groups, n, dim))
        out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-2)
        return out.reshape(1, self.heads, rows, dim).transpose(1, 2)

    def __call__(self, x: torch.Tensor, cache: KVCache, frequencies: torch.Tensor) -> torch.Tensor:
        rows = x.shape[1]
        projected = self.q_proj(x).reshape(1, rows, self.heads, 2 * self.head_dim)
        queries, gate = projected[..., : self.head_dim], projected[..., self.head_dim :]
        keys = self.k_proj(x).reshape(1, rows, self.kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(1, rows, self.kv_heads, self.head_dim)
        queries = rms_norm(queries, self.q_norm, self.eps).transpose(1, 2)
        keys = rms_norm(keys, self.k_norm, self.eps).transpose(1, 2)
        positions = torch.arange(cache.offset, cache.offset + rows, device=x.device)
        queries = rotate(queries, positions, frequencies)
        cache.append(rotate(keys, positions, frequencies), values.transpose(1, 2))
        output = self.attend(queries, cache.keys, cache.values)
        gate = torch.sigmoid(gate.reshape(1, rows, -1))
        return self.o_proj(output.reshape(1, rows, -1) * gate)


class Layer:
    """One decoder layer: linear or full attention, then the MLP."""

    def __init__(self, w: Weights, text: dict[str, Any], linear: bool):
        self.linear = linear
        self.eps = text["rms_norm_eps"]
        self.input_norm = w["input_layernorm.weight"]
        self.post_norm = w["post_attention_layernorm.weight"]
        if linear:
            self.mixer = GatedDeltaNet(w.at("linear_attn"), text)
        else:
            self.mixer = Attention(w.at("self_attn"), text)
        self.gate = w.linear("mlp.gate_proj")
        self.up = w.linear("mlp.up_proj")
        self.down = w.linear("mlp.down_proj")

    def parameters(self) -> list[torch.Tensor]:
        linears = (self.gate, self.up, self.down)
        own = [self.input_norm, self.post_norm, *self.mixer.parameters()]
        return own + [a for layer in linears for a in layer.parameters()]

    def new_cache(self) -> Any:
        return [None, None] if self.linear else KVCache()

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))

    def __call__(self, x: torch.Tensor, cache: Any, frequencies: torch.Tensor | None):
        normed = rms_norm(x, self.input_norm, self.eps)
        if self.linear:
            h = x + self.mixer(normed, cache)
        else:
            h = x + self.mixer(normed, cache, frequencies)
        return h + self.mlp(rms_norm(h, self.post_norm, self.eps))
