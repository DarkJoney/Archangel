# SPDX-License-Identifier: Apache-2.0
"""Prefill forward of the Qwen3.8 Flash-Next blocks, batch one and text only (PyTorch).

Each block follows the oMLX implementation op by op and in the same dtypes.
The gated delta net and the MoE block are ports of the mlx-vlm classes that
implementation builds on (MIT, Copyright (c) 2025 Prince Canuma). Reference
(plain-op) paths only: no fused kernels.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from ...tensors import from_host, unpack
from ...wire import HostArray
from ..qwen3_5 import gdn
from .attention import Attention, QSACache
from .ops import (
    Weights,
    depthwise_conv,
    row_sum,
    stacked_sum,
    stream_norm,
    take,
    unit_scale,
)


class HyperConnection:
    """Reads one block input from the residual streams and gates its write."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.streams, self.hidden = text.get("hc_count", 4), text["hidden_size"]
        self.eps = text["rms_norm_eps"]
        self.scale = unit_scale(w["hc_norm.weight"]).reshape(self.streams, self.hidden)
        self.down = w.linear("input_mix_weight_down")
        self.up = w.linear("input_mix_weight_up")
        self.inject = w.float32("block_inject_weight")

    def parameters(self) -> list[torch.Tensor]:
        linears = (self.down, self.up)
        own = [self.scale, self.inject]
        return own + [a for layer in linears for a in layer.parameters()]

    def mix(self, up: torch.Tensor, normed: torch.Tensor) -> torch.Tensor:
        """``(up, normed) -> block input``."""
        hidden = self.hidden
        # The host kernel takes this sigmoid in float32 and rounds it once.
        gate = torch.sigmoid(up.float()).to(normed.dtype)
        # The streams add up one by one in the residual dtype.
        mixed = gate[..., :hidden] * normed[..., :hidden]
        for stream in range(1, self.streams):
            lo = stream * hidden
            mixed = mixed + gate[..., lo : lo + hidden] * normed[..., lo : lo + hidden]
        return mixed * (1.0 / self.streams)

    def __call__(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(block input [1, L, hidden], write gates [1, L, streams])."""
        normed = stream_norm(h, self.scale, self.eps)
        mix = F.silu(self.down(normed) / self.streams)
        return self.mix(self.up(mix), normed), self.write_gates(normed)

    def write_gates(self, normed: torch.Tensor) -> torch.Tensor:
        """Per-stream gates [1, L, streams] of the block's residual write.

        They read the inject weights without their 16-bit rounding and take
        their sigmoid in float32, as the host kernel does.
        """
        return self.gates(normed.float() @ self.inject.T, normed.dtype)

    def gates(self, dots: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Write gates from the float32 products of the streams and inject."""
        raw = (dots.to(dtype) / self.streams).float()
        return 2 * torch.sigmoid(raw).to(dtype)

    def write(self, h: torch.Tensor, branch: torch.Tensor, inject: torch.Tensor) -> torch.Tensor:
        """Add a block output to every residual stream, scaled by its gate."""
        return h + (branch[..., None, :] * inject[..., None]).reshape(h.shape)


class GatedDeltaNet:
    """Linear attention: a causal conv and the gated delta recurrence."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.key_heads = text["linear_num_key_heads"]
        self.value_heads = text["linear_num_value_heads"]
        self.key_dim = text["linear_key_head_dim"]
        self.value_dim = text["linear_value_head_dim"]
        self.eps = text["rms_norm_eps"]
        self.sigmoid_gate = (
            text.get("output_gate_type") or text.get("hidden_act") or "sigmoid"
        ) == "sigmoid"
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

    def prework(self, qkv: torch.Tensor, state: torch.Tensor | None):
        """(q, k, v, next conv window) from the projected rows [1, rows, C]."""
        rows, window = qkv.shape[1], self.taps.shape[1] - 1
        if state is None:
            state = torch.zeros(1, window, qkv.shape[-1], dtype=qkv.dtype, device=qkv.device)
        conv_in = torch.cat([state, qkv], dim=1)
        conv_out = F.silu(depthwise_conv(conv_in, self.taps, 1))
        keys = self.key_heads * self.key_dim
        q = conv_out[..., :keys].reshape(1, rows, self.key_heads, self.key_dim)
        k = conv_out[..., keys : 2 * keys].reshape(q.shape)
        v = conv_out[..., 2 * keys :]
        v = v.reshape(1, rows, self.value_heads, self.value_dim)
        # L2 norm with the epsilon after the sum; the query takes 1/sqrt(dim).
        q = q * torch.rsqrt(row_sum(torch.square(q)) + 1e-6)
        k = k * torch.rsqrt(row_sum(torch.square(k)) + 1e-6)
        return q * self.key_dim**-0.5, k, v, conv_in[:, rows:]

    def gates(self, a: torch.Tensor, b: torch.Tensor):
        """(decay in float32, update rate) of each step and value head."""
        return torch.exp(-self.decay * F.softplus(a + self.dt_bias).float()), torch.sigmoid(b)

    def finish(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """The recurrence output normalized and gated, [1, rows, value width]."""
        squares = torch.square(y.float())
        inverse = torch.rsqrt(squares.mean(dim=-1, keepdim=True) + self.eps)
        weight = self.norm_weight
        # The norm rounds before its weight; the gate product is float32.
        normed = (y.float() * inverse).to(weight.dtype) * weight
        z = z.reshape(y.shape).float()
        act = torch.sigmoid(z) if self.sigmoid_gate else F.silu(z)
        gated = (normed.float() * act).to(weight.dtype)
        return gated.reshape(1, y.shape[1], -1)

    def __call__(self, x: torch.Tensor, cache: list) -> torch.Tensor:
        q, k, v, cache[0] = self.prework(self.qkv(x), cache[0])
        gamma, beta = self.gates(self.a(x), self.b(x))
        y, cache[1] = gdn.update(q, k, v, gamma, beta, cache[1])
        return self.out(self.finish(y, self.z(x)))


def ple_rows(inputs: dict[str, HostArray], name: str, rows: int) -> torch.Tensor:
    """The chunk's n-gram embeddings [1, rows, dim] from the rows the host read."""
    ids = inputs.get(name + "ids")
    if ids is None or ids[0].shape[1] != rows:
        raise RuntimeError(f"No PLE rows for this chunk ({name}ids)")
    data = [from_host(inputs[f"{name}{n}"]) for n in range(3) if f"{name}{n}" in inputs]
    scale = from_host(inputs[name + "ws"])
    bits, group_size, cast, fp8 = (int(v) for v in inputs[name + "q"][0])
    values = data[0]
    if fp8:
        raise NotImplementedError("fp8 PLE rows on the XPU")
    if bits:
        values = unpack(values, data[1], data[2], {"group_size": group_size, "bits": bits})
    if cast:
        values = values.to(scale.dtype)
    return (values * scale).reshape(1, rows, -1)


class PLE:
    """N-gram embedding block; the host owns the table and sends the rows."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.streams, self.hidden = text.get("hc_count", 4), text["hidden_size"]
        self.eps = text["rms_norm_eps"]
        shape = (self.streams, self.hidden)
        self.key = w.linear("key_proj")
        self.value = w.linear("value_proj")
        self.key_scale = unit_scale(w["norm_key.weight"]).reshape(shape)
        self.query_scale = unit_scale(w["norm_query.weight"]).reshape(shape)
        self.conv_scale = unit_scale(w["norm_conv.weight"]).reshape(shape)
        self.taps = w["conv1d.weight"][..., 0].float()
        ngram = text.get("ngram_size", 3)
        self.dilation, self.context = ngram, ngram - 1
        self.window = (self.taps.shape[1] - 1) * self.dilation
        self.dim = text.get("ple_embed_dim") or self.hidden
        eos = text.get("eos_token_id")
        self.eos = eos[0] if isinstance(eos, list) else eos

    def parameters(self) -> list[torch.Tensor]:
        own = [self.key_scale, self.query_scale, self.conv_scale, self.taps]
        return own + self.key.parameters() + self.value.parameters()

    def __call__(
        self, h: torch.Tensor, tokens: torch.Tensor, rows: torch.Tensor | None, cache: list
    ) -> torch.Tensor:
        """``rows`` is None only for a warm-up chunk, which reads a zero table."""
        length = h.shape[1]
        history = cache[3]
        if history is None:
            history = torch.full((1, self.context), self.eos, dtype=torch.int64, device=h.device)
        history = torch.cat([history, tokens.long()[None]], dim=-1)
        cache[3] = history[:, length:]

        if rows is None:
            rows = torch.zeros(1, length, self.dim, dtype=h.dtype, device=h.device)
        embeddings = rows.to(h.dtype)
        keys = stream_norm(self.key(embeddings), self.key_scale, self.eps)
        queries = stream_norm(h, self.query_scale, self.eps)
        gated = self.gated(keys, queries, self.value(embeddings))
        normed = stream_norm(gated, self.conv_scale, self.eps)
        conv, cache[2] = self.convolved(normed, cache[2])
        return gated + F.silu(conv)

    def gated(self, keys: torch.Tensor, queries: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """The value gated per stream by the match of its key and the query."""
        shape = (1, keys.shape[1], self.streams, self.hidden)
        gate = row_sum(keys.reshape(shape) * queries.reshape(shape))
        gate = gate / math.sqrt(self.hidden)
        gate = torch.sign(gate) * torch.sqrt(torch.clamp(gate.abs(), min=1e-6))
        return (torch.sigmoid(gate) * value[..., None, :]).reshape(keys.shape)

    def convolved(self, normed: torch.Tensor, state: torch.Tensor | None):
        """(dilated conv of the rows, next conv window)."""
        if state is None:
            state = torch.zeros(
                1, self.window, normed.shape[-1], dtype=normed.dtype, device=normed.device
            )
        conv_in = torch.cat([state, normed], dim=1)
        return (
            depthwise_conv(conv_in, self.taps, self.dilation),
            conv_in[:, normed.shape[1] :],
        )


# Experts unpacked together (bounds the dense copy: block * 3 * inner * hidden * 4 B).
_EXPERT_BLOCK = 16


class Bank:
    """Expert projections of one input, joined along their outputs.

    Dense banks are stored [experts, out, in]. A quantized bank is unpacked one
    expert at a time (an expert bank would not fit unpacked).
    """

    def __init__(self, w: Weights, names: tuple[str, ...]):
        if f"{names[0]}.scales" not in w:
            self.dense = torch.cat([w[f"{n}.weight"] for n in names], dim=1)
            self.packed, self.quant, self.dtype = None, None, self.dense.dtype
            self.experts = self.dense.shape[0]
            self.device = self.dense.device
        else:
            self.dense = None
            self.packed = [
                torch.cat([w[f"{n}.{part}"] for n in names], dim=1)
                for part in ("weight", "scales", "biases")
            ]
            self.quant = w.quantization(names[0])
            self.dtype = self.packed[1].dtype
            self.experts = self.packed[0].shape[0]
            self.device = self.packed[0].device

    def parameters(self) -> list[torch.Tensor]:
        return [self.dense] if self.packed is None else list(self.packed)

    def weights(self, experts: list[int]) -> torch.Tensor:
        """The dense [n, out, in] weights of some experts, unpacked in one go."""
        index = torch.tensor(experts, dtype=torch.int64, device=self.device)
        if self.packed is None:
            return torch.index_select(self.dense, 0, index)
        weight, scales, biases = (torch.index_select(p, 0, index) for p in self.packed)
        return unpack(weight, scales, biases, self.quant, self.dtype)


class MoE:
    """Routed expert MLPs plus a gated shared expert."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.top_k = text["num_experts_per_tok"]
        self.inner = text["moe_intermediate_size"]
        self.normalize = text.get("norm_topk_prob", True)
        self.router = w.linear("gate")
        self.shared_gate = w.linear("shared_expert.gate_proj")
        self.shared_up = w.linear("shared_expert.up_proj")
        self.shared_down = w.linear("shared_expert.down_proj")
        self.shared_scale = w.linear("shared_expert_gate")
        banks = w.at("switch_mlp")
        # Gate and up share their input rows: one product computes both.
        self.gate_up = Bank(banks, ("gate_proj", "up_proj"))
        self.down = Bank(banks, ("down_proj",))

    def parameters(self) -> list[torch.Tensor]:
        linears = (
            self.router,
            self.shared_gate,
            self.shared_up,
            self.shared_down,
            self.shared_scale,
        )
        return self.gate_up.parameters() + self.down.parameters() + [
            a for layer in linears for a in layer.parameters()
        ]

    def route(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(expert ids, weights) [1, rows, top_k] from the router logits."""
        k = self.top_k
        gates = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        scores, chosen = torch.topk(gates, k, dim=-1)
        if self.normalize:
            # The host adds these few values one by one in their dtype.
            total = scores[..., :1]
            for i in range(1, k):
                total = scores[..., i : i + 1] + total
            scores = scores / total
        return chosen, scores

    def experts(self, x: torch.Tensor, chosen: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expert outputs [rows * top_k, hidden] sorted by expert, and the row
        of each (token, choice) in them."""
        k = self.top_k
        flat = chosen.reshape(-1)
        order = torch.argsort(flat, stable=True)
        ids = take(flat, order)
        rows = x.reshape(x.shape[1], -1)
        routed = take(rows, order // k)
        out = torch.empty(routed.shape[0], rows.shape[1], dtype=x.dtype, device=x.device)
        used, counts = torch.unique_consecutive(ids, return_counts=True)
        used, counts = used.tolist(), counts.tolist()
        start = 0
        # Unpack the experts a block at a time: few launches, bounded memory.
        for first in range(0, len(used), _EXPERT_BLOCK):
            block = used[first : first + _EXPERT_BLOCK]
            gate_up = self.gate_up.weights(block)
            down = self.down.weights(block)
            for j, count in enumerate(counts[first : first + _EXPERT_BLOCK]):
                part = routed[start : start + count]
                both = F.linear(part, gate_up[j])
                hidden = F.silu(both[:, : self.inner]) * both[:, self.inner :]
                out[start : start + count] = F.linear(hidden, down[j])
                start += count
        return out, torch.argsort(order)

    def combined(self, y: torch.Tensor, positions: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Each token's expert outputs times their weights, summed."""
        rows, k = scores.shape[1:]
        picked = take(y, positions).reshape(1, rows, k, -1)
        return stacked_sum(picked * scores[..., None])

    def shared(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.shared_down(F.silu(self.shared_gate(x)) * self.shared_up(x))
        return torch.sigmoid(self.shared_scale(x)) * shared

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        chosen, scores = self.route(self.router(x))
        return self.combined(*self.experts(x, chosen), scores) + self.shared(x)


class Layer:
    """One decoder layer: optional PLE, attention and MoE on the streams."""

    def __init__(self, w: Weights, text: dict[str, Any], kind: str, ple: bool):
        self.ple = PLE(w.at("ple"), text) if ple else None
        self.attn_hc = HyperConnection(w.at("attn_hyper_connection"), text)
        self.mlp_hc = HyperConnection(w.at("mlp_hyper_connection"), text)
        self.qsa = kind == "qsa"
        if self.qsa:
            self.mixer = Attention(w.at("self_attn"), text)
        else:
            self.mixer = GatedDeltaNet(w.at("linear_attn"), text)
        self.mlp = MoE(w.at("mlp"), text)

    def parameters(self) -> list[torch.Tensor]:
        blocks = [self.attn_hc, self.mlp_hc, self.mixer, self.mlp]
        if self.ple is not None:
            blocks.append(self.ple)
        return [a for block in blocks for a in block.parameters()]

    def new_cache(self) -> Any:
        if self.qsa:
            return QSACache()
        return [None] * (4 if self.ple is not None else 2)

    def __call__(
        self, h: torch.Tensor, tokens: torch.Tensor, rows: torch.Tensor | None, cache: Any
    ) -> torch.Tensor:
        if self.ple is not None:
            h = h + self.ple(h, tokens, rows, cache)
        mixed, inject = self.attn_hc(h)
        h = self.attn_hc.write(h, self.mixer(mixed, cache), inject)
        mixed, inject = self.mlp_hc(h)
        return self.mlp_hc.write(h, self.mlp(mixed), inject)
