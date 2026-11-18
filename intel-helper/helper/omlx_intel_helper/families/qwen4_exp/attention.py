# SPDX-License-Identifier: Apache-2.0
"""Qwen sparse attention (QSA) for prefill, batch one and text only (PyTorch).

A query that sees more than ``top_k`` complete blocks of ``ratio`` keys attends
to its ``top_k`` best blocks and the incomplete tail, others to every key.

The rotary embedding and the selection mask are ports from mlx-vlm (MIT,
Copyright (c) 2025 Prince Canuma), the gathered arm from oMLX. Reference
(plain-op) paths only.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ...tensors import DEVICE
from ..qwen3_5.layers import rotate
from .ops import Weights, norm, take, unit_scale

# Chunks shorter than this stay on masked dense attention on the host.
GATHERED_MIN_QUERIES = 16
# Query rows of one gathered reference step; bounds the gathered key copy.
_REFERENCE_ROWS = 128


class QSACache:
    """Keys, values and indexer state of one QSA layer for one prompt."""

    def __init__(self) -> None:
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.index_keys: torch.Tensor | None = None
        self.positions: torch.Tensor | None = None
        # Normalized, rotated mean key of each complete block.
        self.pooled: torch.Tensor | None = None
        self.offset = 0
        # (keys, values, index keys, positions) of the last chunk.
        self.appended: tuple[torch.Tensor, ...] = ()

    def append(self, keys, values, index_keys, positions) -> None:
        """Add rows: keys/values [1, heads, n, dim], index keys [1, n, dim]."""
        self.appended = (keys, values, index_keys, positions)
        if self.keys is None:
            self.keys, self.values = keys, values
            self.index_keys, self.positions = index_keys, positions
        else:
            self.keys = torch.cat([self.keys, keys], dim=2)
            self.values = torch.cat([self.values, values], dim=2)
            self.index_keys = torch.cat([self.index_keys, index_keys], dim=1)
            self.positions = torch.cat([self.positions, positions], dim=1)
        self.offset = self.keys.shape[2]

    def arrays(self) -> list[torch.Tensor]:
        held = (self.keys, self.values, self.index_keys, self.positions, self.pooled)
        return [a for a in held if a is not None]


class Attention:
    """Gated multi-head attention over the QSA selection of cached tokens."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.heads = text["num_attention_heads"]
        self.kv_heads = text["num_key_value_heads"]
        self.head_dim = text.get("head_dim") or text["hidden_size"] // self.heads
        self.eps = text["rms_norm_eps"]
        self.q_proj = w.linear("q_proj")
        self.k_proj = w.linear("k_proj")
        self.v_proj = w.linear("v_proj")
        self.o_proj = w.linear("o_proj")
        self.q_scale = unit_scale(w["q_norm.weight"])
        self.k_scale = unit_scale(w["k_norm.weight"])

        self.index_heads = text.get("indexer_n_heads", 4)
        self.index_dim = text.get("indexer_head_dim", 128)
        self.budget = text.get("indexer_budget", 2048)
        self.ratio = text.get("indexer_compress_ratio", 4)
        self.top_k = self.budget // self.ratio
        self.index_proj = w.linear("indexer.index_qk_proj")
        self.index_q_scale = unit_scale(w["indexer.q_layernorm.weight"])
        self.index_k_scale = unit_scale(w["indexer.k_layernorm.weight"])

        rope = text.get("rope_parameters") or {}
        rotary = int(self.head_dim * rope.get("partial_rotary_factor", 0.25))
        if rotary > min(self.head_dim, self.index_dim):
            raise ValueError("The rotary width exceeds an attention head")
        # Float32 on the CPU: every helper build rotates by the same angles.
        exponent = np.arange(0, rotary, 2, dtype=np.float32) / np.float32(rotary)
        base = np.float32(rope.get("rope_theta", 10_000_000))
        inv = (np.float32(1.0) / base**exponent).astype(np.float32)
        self.inv_freq = torch.from_numpy(inv).to(DEVICE)

    def parameters(self) -> list[torch.Tensor]:
        linears = (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.index_proj)
        scales = [self.q_scale, self.k_scale, self.index_q_scale, self.index_k_scale]
        return scales + [a for layer in linears for a in layer.parameters()]

    def _pooled(self, cache: QSACache) -> torch.Tensor:
        """The block bank [1, blocks, dim], extended by the new complete blocks."""
        ratio = self.ratio
        have = 0 if cache.pooled is None else cache.pooled.shape[1]
        complete = cache.offset // ratio
        if have < complete:
            raw = cache.index_keys[:, have * ratio : complete * ratio]
            raw = raw.reshape(1, complete - have, ratio, self.index_dim)
            pooled = raw.float().mean(dim=-2).to(raw.dtype)
            pooled = norm(pooled, self.index_k_scale, self.eps)
            # A block is rotated by the position of its first token.
            starts = cache.positions[0, have * ratio : complete * ratio : ratio]
            pooled = rotate(pooled[:, None], starts, self.inv_freq)[:, 0]
            if cache.pooled is not None:
                pooled = torch.cat([cache.pooled, pooled], dim=1)
            cache.pooled = pooled
        return cache.pooled

    def _block_scores(self, index_queries: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        """Float32 score [rows, blocks] of each block for each query.

        Float32 products, heads summed in host order; rounding would change the picks.
        """
        rows, blocks = index_queries.shape[0], pooled.shape[1]
        scores = index_queries.float().reshape(rows * self.index_heads, self.index_dim) @ (
            pooled[0].float().transpose(-1, -2)
        )
        scores = scores.reshape(rows, self.index_heads, blocks)
        out = torch.clamp(scores[:, 0], min=0)
        for head in range(1, self.index_heads):
            out = out + torch.clamp(scores[:, head], min=0)
        return out / math.sqrt(self.index_dim)

    def _select(
        self, index_queries: torch.Tensor, pooled: torch.Tensor, first_visible: int
    ) -> torch.Tensor:
        """Each query's selected blocks in key order, [rows, width] int64.

        ``index_queries`` [rows, heads, dim] are normalized and rotated and
        row ``i`` sees ``first_visible + i`` tokens. A query with at most
        ``top_k`` complete blocks gets the first ``width`` block ids.
        """
        rows, blocks = index_queries.shape[0], pooled.shape[1]
        dev = index_queries.device
        if blocks <= self.top_k:
            first = torch.arange(blocks, dtype=torch.int64, device=dev)[None]
            return first.expand(rows, blocks)
        scores = self._block_scores(index_queries, pooled)
        complete = (first_visible + torch.arange(rows, dtype=torch.int64, device=dev)) // self.ratio
        valid = torch.arange(blocks, device=dev)[None] < complete[:, None]
        scores = scores.masked_fill(~valid, float("-inf"))
        # The best top_k by a stable ascending sort, the last ones: among tied
        # scores (a block no head likes scores exactly 0) the later blocks win,
        # as in the host's argpartition.
        ranked = torch.argsort(scores, dim=-1, stable=True)[..., -self.top_k :]
        first = torch.arange(self.top_k, dtype=torch.int64, device=dev)[None]
        ranked = torch.where((complete <= self.top_k)[:, None], first, ranked)
        return torch.sort(ranked, dim=-1).values

    def _selection_mask(self, selected: torch.Tensor, visible: torch.Tensor, keys: int) -> torch.Tensor:
        """Boolean attention mask [rows, keys] of a block selection."""
        rows, blocks = selected.shape[0], keys // self.ratio
        dev = selected.device
        hits = torch.zeros(rows, blocks, dtype=torch.bool, device=dev)
        hits.scatter_(1, selected.long(), True)
        tokens = torch.repeat_interleave(hits, self.ratio, dim=-1)
        if blocks * self.ratio < keys:
            pad = torch.zeros(rows, keys - blocks * self.ratio, dtype=torch.bool, device=dev)
            tokens = torch.cat([tokens, pad], dim=-1)
        complete = visible // self.ratio
        token = torch.arange(keys, device=dev)[None]
        causal = token < visible[:, None]
        tail = (token >= (complete * self.ratio)[:, None]) & causal
        sparse = (complete > self.top_k)[:, None]
        return torch.where(sparse, tokens | tail, causal)

    def _gathered(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        selected: torch.Tensor,
        visible: torch.Tensor,
    ) -> torch.Tensor:
        """Attention of [1, heads, rows, dim] queries over their selected keys.

        Returns [1, rows, heads, dim]. The scores and the softmax are float32.
        """
        ratio, rows, width = self.ratio, queries.shape[2], selected.shape[1]
        dev = queries.device
        groups = self.heads // self.kv_heads
        complete = visible // ratio
        count = torch.clamp(complete, max=self.top_k)
        tokens = selected.long()[..., None] * ratio + torch.arange(ratio, device=dev)
        valid = torch.repeat_interleave(
            torch.arange(width, device=dev)[None] < count[:, None], ratio, dim=-1
        )
        tail = complete[:, None] * ratio + torch.arange(ratio - 1, device=dev)
        tokens = torch.cat([tokens.reshape(rows, width * ratio), tail], dim=-1)
        valid = torch.cat([valid, tail < visible[:, None]], dim=-1)
        flat = torch.where(valid, tokens, torch.zeros_like(tokens)).reshape(-1)

        def gather(cached: torch.Tensor) -> torch.Tensor:
            picked = take(cached[0].transpose(0, 1), flat)  # [rows * S, kv heads, dim]
            return picked.reshape(rows, -1, self.kv_heads, self.head_dim).permute(0, 2, 1, 3)

        grouped = queries[0].transpose(0, 1).reshape(rows, self.kv_heads, groups, self.head_dim)
        scores = (grouped.float() @ gather(keys).float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        out = torch.softmax(scores, dim=-1) @ gather(values).float()
        return out.to(queries.dtype).reshape(1, rows, self.heads, self.head_dim)

    def _masked(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Attention over the keys a boolean ``mask`` [rows, keys] allows.

        Returns [1, rows, heads, dim]. Scores and softmax are float32, like
        the gathered rows: a 16-bit score sheet moves the softmax weights.
        """
        rows = queries.shape[2]
        grouped = queries.reshape(1, self.kv_heads, -1, rows, self.head_dim)
        scores = (grouped.float() @ keys[:, :, None].float().transpose(-1, -2)) / math.sqrt(
            self.head_dim
        )
        scores = scores.masked_fill(~mask, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ values[:, :, None].float()
        out = out.to(queries.dtype).reshape(1, self.heads, rows, self.head_dim)
        return out.transpose(1, 2)

    def _sparse_rows(
        self,
        queries: torch.Tensor,
        index_queries: torch.Tensor,
        cache: QSACache,
        first: int,
    ) -> torch.Tensor:
        """Gathered attention of the query rows from ``first`` to the end."""
        rows = queries.shape[2]
        pooled = self._pooled(cache)
        outputs = []
        for start in range(first, rows, _REFERENCE_ROWS):
            stop = min(start + _REFERENCE_ROWS, rows)
            seen = cache.offset - rows + start + 1
            visible = seen + torch.arange(stop - start, dtype=torch.int64, device=queries.device)
            selected = self._select(index_queries[start:stop], pooled, seen)
            part = queries[:, :, start:stop]
            outputs.append(self._gathered(part, cache.keys, cache.values, selected, visible))
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)

    def prepare(self, q_out, k_out, v_out, index, positions):
        """Normalized, rotated operands from the four projections of a chunk.

        Returns (queries [1, heads, rows, dim], output gate [1, rows, -1],
        keys and values [1, kv heads, rows, dim], raw index keys
        [1, rows, index dim], index queries [rows, index heads, index dim]).
        """
        rows = q_out.shape[1]
        queries, gate = q_out.reshape(1, rows, self.heads, -1).chunk(2, dim=-1)
        queries = norm(queries, self.q_scale, self.eps).transpose(1, 2)
        keys = norm(k_out.reshape(1, rows, self.kv_heads, -1), self.k_scale, self.eps)
        keys = keys.transpose(1, 2)
        values = v_out.reshape(1, rows, self.kv_heads, -1).transpose(1, 2)
        index = index.reshape(rows, self.index_heads + 1, self.index_dim)
        index_queries = norm(index[:, : self.index_heads], self.index_q_scale, self.eps)
        # Rotated like the block bank.
        index_queries = rotate(index_queries.transpose(0, 1)[None], positions, self.inv_freq)
        return (
            rotate(queries, positions, self.inv_freq),
            gate.reshape(1, rows, -1),
            rotate(keys, positions, self.inv_freq),
            values,
            index[None, :, self.index_heads],
            index_queries[0].transpose(0, 1),
        )

    def __call__(self, x: torch.Tensor, cache: QSACache) -> torch.Tensor:
        _, rows, _ = x.shape
        dev = x.device
        past = cache.offset
        positions = torch.arange(past, past + rows, dtype=torch.int32, device=dev)
        outs = (self.q_proj(x), self.k_proj(x), self.v_proj(x), self.index_proj(x))
        queries, gate, keys, values, index_keys, index_queries = self.prepare(*outs, positions)
        cache.append(keys, values, index_keys, positions[None])

        total = past + rows
        token = torch.arange(total, device=dev)[None]
        if rows >= GATHERED_MIN_QUERIES and total > self.budget:
            # Rows that see at most top_k complete blocks attend to all keys.
            dense = min(rows, max(0, self.budget + self.ratio - 1 - past))
            if rows - dense == 1:
                dense -= 1
            outputs = []
            if dense:
                rows_dense = past + torch.arange(dense, device=dev)[:, None]
                causal = token[:, : past + dense] <= rows_dense
                outputs.append(
                    self._masked(
                        queries[:, :, :dense],
                        cache.keys[:, :, : past + dense],
                        cache.values[:, :, : past + dense],
                        causal,
                    )
                )
            if dense < rows:
                outputs.append(self._sparse_rows(queries, index_queries, cache, dense))
            output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=1)
        else:
            visible = past + torch.arange(rows, dtype=torch.int64, device=dev) + 1
            if total // self.ratio > self.top_k:
                selected = self._select(index_queries, self._pooled(cache), past + 1)
                mask = self._selection_mask(selected, visible, total)
            else:
                mask = token < visible[:, None]
            output = self._masked(queries, cache.keys, cache.values, mask)
        return self.o_proj(output.reshape(1, rows, -1) * torch.sigmoid(gate))
