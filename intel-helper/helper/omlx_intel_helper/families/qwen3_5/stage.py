# SPDX-License-Identifier: Apache-2.0
"""One GPU stage of Qwen3.5: layers ``[first, last)`` (PyTorch XPU).

The boundary of every cut is the hidden state ``h`` [1, L, hidden], with the
host's rotary frequencies ``rope`` when the job runs an attention layer on the
helper. A linear layer's state is its conv window and recurrent state, sent
whole; an attention layer's state is the K/V rows the chunk appended.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ... import checkpoint
from ...tensors import (
    DEVICE,
    allocated,
    device_name,
    empty_cache,
    from_host,
    load,
    peak_bytes,
    synchronize,
    to_host,
    unpack,
)
from ...wire import HostArray
from . import check_layout, linear_layers, resident
from . import text_config as read_text_config
from .layers import KVCache, Layer, Weights, rotary_frequencies

Arrays = dict[str, HostArray]


class Stage:
    def __init__(self, model_dir: str | Path, first: int, last: int):
        config = checkpoint.load_config(model_dir)
        text = read_text_config(config)
        self.first, self.last = first, last
        self.device = device_name()
        self.hidden = text["hidden_size"]
        self.vocab = text["vocab_size"]

        index = checkpoint.tensors(model_dir)
        check_layout(index)
        root = "language_model.model."
        names = [
            name
            for name in index
            if resident(name)
            and (
                first <= layer < last
                if (layer := checkpoint.layer_of(name)) is not None
                else first == 0
            )
        ]
        linear = linear_layers(text)

        def load_layer(i: int) -> Layer:
            wanted = [n for n in names if n.startswith(f"{root}layers.{i}.")]
            if not wanted:  # a layer added back after a drop has its files too
                wanted = [
                    n for n in index if n.startswith(f"{root}layers.{i}.") and resident(n)
                ]
            weights = Weights(load(model_dir, wanted), config, root)
            layer = Layer(weights.at(f"layers.{i}"), text, linear[i])
            # The packed tensors of the unpacked linears are no longer needed.
            del weights
            empty_cache()
            return layer

        self._load_layer = load_layer
        self.layers = [load_layer(i) for i in range(first, last)]
        self.embedding = None
        if first == 0:
            wanted = [n for n in names if checkpoint.layer_of(n) is None]
            weights = Weights(load(model_dir, wanted), config, root)
            self.embedding = [
                weights.arrays[f"{root}embed_tokens.{part}"]
                for part in ("weight", "scales", "biases")
                if f"{root}embed_tokens.{part}" in weights.arrays
            ]
            self.embedding_quant = checkpoint.quantization(
                config, f"{root}embed_tokens"
            )
        self.dtype = self.layers[0].input_norm.dtype
        self.frequencies = rotary_frequencies(text, DEVICE)

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        # index_select, not part[tokens]: advanced indexing of a tensor with a
        # 2-element row crashes this torch XPU build (access violation).
        weight, *quantized = (
            torch.index_select(part, 0, tokens) for part in self.embedding
        )
        if not quantized:
            return weight
        return unpack(weight, *quantized, self.embedding_quant, self.dtype)

    def memory_bytes(self) -> int:
        return allocated()

    def peak_bytes(self) -> int:
        return peak_bytes()

    def drop_layer(self) -> None:
        """Free the weights of the last layer held."""
        self.layers.pop()
        self.last -= 1
        empty_cache()

    def add_layer(self) -> None:
        """Load the layer after the last one held, as the stage loaded it."""
        self.layers.append(self._load_layer(self.last))
        self.last += 1

    def new_job(self, layers: int) -> list:
        return [layer.new_cache() for layer in self.layers[:layers]]

    def trim(self, job: list, layers: int) -> None:
        del job[layers:]

    def load_prefix(self, job: list, state: Arrays) -> None:
        for j, cache in enumerate(job):
            name = f"{self.first + j}."
            if isinstance(cache, KVCache):
                if name + "k" in state:
                    cache.append(
                        from_host(state[name + "k"]), from_host(state[name + "v"])
                    )
            else:
                for slot in range(len(cache)):
                    if f"{name}{slot}" in state:
                        cache[slot] = from_host(state[f"{name}{slot}"])

    def _forward(
        self,
        job: list,
        h: torch.Tensor,
        frequencies: torch.Tensor | None,
        keep: dict[int, int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[int, torch.Tensor]]:
        """Run the job's layers on ``h``.

        Returns the hidden state, the chunk's state and, for each layer in
        ``keep``, its output from that chunk row on.
        """
        state: dict[str, torch.Tensor] = {}
        kept: dict[int, torch.Tensor] = {}
        for j, cache in enumerate(job):
            layer, index = self.layers[j], self.first + j
            if not layer.linear and frequencies is None:
                raise RuntimeError("No rotary frequencies for this chunk (rope)")
            h = layer(h, cache, frequencies)
            if layer.linear:
                state[f"{index}.0"], state[f"{index}.1"] = cache
            else:
                state[f"{index}.k"], state[f"{index}.v"] = cache.appended
            if keep and index in keep:
                kept[index] = h[:, keep[index] :]
        return h, state, kept

    def enable_kernels(self) -> None:
        """No custom kernels on the XPU yet (kept for the stage contract)."""

    @torch.inference_mode()
    def warm(self, tokens: int) -> float:
        ids = torch.arange(tokens, device=DEVICE) % (self.vocab - 3) + 3
        if self.first:
            h = torch.randn(1, tokens, self.hidden, device=DEVICE).to(self.dtype)
        else:
            h = self._embed(ids)[None]
        seconds = 0.0
        # The first pass compiles kernels; the second one is the pace.
        for _ in range(2):
            began = time.perf_counter()
            self._forward(self.new_job(len(self.layers)), h, self.frequencies)
            synchronize()
            seconds = time.perf_counter() - began
        empty_cache()
        return seconds

    @torch.inference_mode()
    def run(
        self,
        job: list,
        *,
        start: int,
        tokens: np.ndarray,
        boundary: Arrays | None,
        inputs: Arrays,
        following: int,
        capture: dict | None,
    ) -> tuple[Arrays, Arrays]:
        # The host's frequencies reach the first stage as an input and every
        # later stage with the hidden state.
        if boundary is None:
            ids = torch.from_numpy(np.asarray(tokens, dtype=np.int64)).to(DEVICE)
            h = self._embed(ids)[None]
            rope = inputs.get("rope")
        else:
            h = from_host(boundary["h"])
            rope = boundary.get("rope")
        frequencies = None if rope is None else from_host(rope)
        if rope is not None and frequencies.shape != self.frequencies.shape:
            raise ValueError("The rotary frequencies do not fit this model (rope)")
        keep = None
        if capture and start + h.shape[1] > capture["from"]:
            row = max(0, capture["from"] - start)
            keep = {i: row for i in capture["layers"]}
        h, state, kept = self._forward(job, h, frequencies, keep)
        out = {f"cap.{i}": to_host(rows) for i, rows in kept.items()}
        out["h"] = to_host(h)
        if rope is not None:
            out["rope"] = rope
        return out, {name: to_host(a) for name, a in state.items()}


def load_stage(model_dir: str | Path, first: int, last: int) -> Stage:
    return Stage(model_dir, first, last)
