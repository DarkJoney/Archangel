# SPDX-License-Identifier: Apache-2.0
"""One GPU stage of Qwen3.8 Flash-Next: layers ``[first, last)`` (PyTorch).

The boundary of every cut is the residual streams with all writes applied,
``h`` [1, L, streams * hidden]. A linear layer's state is its conv window and
recurrent state (the PLE layer adds its conv window and token history), sent
whole; a QSA layer's state is the rows the chunk appended.
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
from . import (
    FIRST_STAGE_MIN,
    check_first_stage,
    layer_kinds,
    ple_layers,
    resident,
)
from . import text_config as read_text_config
from .attention import QSACache
from .layers import Layer, ple_rows
from .ops import Weights

Arrays = dict[str, HostArray]
_QSA_SLOTS = ("k", "v", "ik", "ip")


class Stage:
    def __init__(self, model_dir: str | Path, first: int, last: int):
        config = checkpoint.load_config(model_dir)
        text = read_text_config(config)
        check_first_stage(text)
        if 0 < first < FIRST_STAGE_MIN:
            raise ValueError(f"A later stage starts at layer {FIRST_STAGE_MIN} or above")
        self.first, self.last = first, last
        self.device = device_name()
        self.streams = text.get("hc_count", 4)
        self.hidden = text["hidden_size"]
        self.vocab = text["vocab_size"]

        index = checkpoint.tensors(model_dir)
        embedding = next(n for n in index if n.endswith("embed_tokens.weight"))
        root = embedding.removesuffix("embed_tokens.weight")
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
        kinds, ple = layer_kinds(text), ple_layers(text)

        def load_layer(i: int) -> Layer:
            wanted = [n for n in names if n.startswith(f"{root}layers.{i}.")]
            if not wanted:
                wanted = [n for n in index if n.startswith(f"{root}layers.{i}.") and resident(n)]
            weights = Weights(load(model_dir, wanted), config, root)
            layer = Layer(weights.at(f"layers.{i}"), text, kinds[i], i in ple)
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
                weights[f"embed_tokens.{part}"]
                for part in ("weight", "scales", "biases")
                if f"embed_tokens.{part}" in weights
            ]
            self.embedding_quant = weights.quantization("embed_tokens")
            self.dtype = self._embed(torch.zeros(1, dtype=torch.int64, device=DEVICE)).dtype
        else:
            self.dtype = self.layers[0].attn_hc.down.weight.dtype

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        # index_select, not part[tokens]: advanced indexing of a tensor with a
        # 2-element row crashes some torch XPU builds (access violation).
        weight, *quantized = (torch.index_select(part, 0, tokens) for part in self.embedding)
        if not quantized:
            return weight
        return unpack(weight, *quantized, self.embedding_quant)

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
            if isinstance(cache, QSACache):
                if name + "k" in state:
                    cache.append(*(from_host(state[name + s]) for s in _QSA_SLOTS))
            else:
                for slot in range(len(cache)):
                    if f"{name}{slot}" in state:
                        cache[slot] = from_host(state[f"{name}{slot}"])

    def _forward(
        self,
        job: list,
        tokens: torch.Tensor,
        h: torch.Tensor | None,
        inputs: Arrays | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the job's layers; ``inputs`` is None only for a warm-up chunk."""
        if h is None:
            h = self._embed(tokens)[None].repeat(1, 1, self.streams)
        state: dict[str, torch.Tensor] = {}
        for j, cache in enumerate(job):
            layer, index = self.layers[j], self.first + j
            rows = None
            if layer.ple is not None and inputs is not None:
                rows = ple_rows(inputs, f"p{index}.", tokens.shape[0])
            h = layer(h, tokens, rows, cache)
            if isinstance(cache, QSACache):
                state.update({f"{index}.{s}": a for s, a in zip(_QSA_SLOTS, cache.appended)})
            else:
                state.update({f"{index}.{s}": a for s, a in enumerate(cache) if a is not None})
        return h, state

    def enable_kernels(self) -> None:
        """No custom kernels on the XPU yet (kept for the stage contract)."""

    @torch.inference_mode()
    def warm(self, tokens: int) -> float:
        ids = torch.arange(tokens, device=DEVICE, dtype=torch.int64) % (self.vocab - 3) + 3
        h = None
        if self.first:
            shape = (1, tokens, self.streams * self.hidden)
            h = torch.randn(shape, device=DEVICE).to(self.dtype)
        seconds = 0.0
        # The first pass compiles kernels; the second one is the pace.
        for _ in range(2):
            began = time.perf_counter()
            self._forward(self.new_job(len(self.layers)), ids, h, None)
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
        # The host adapter of this family captures no layer outputs.
        h = None if boundary is None else from_host(boundary["h"])
        ids = torch.from_numpy(np.asarray(tokens, dtype=np.int64)).to(DEVICE)
        h, state = self._forward(job, ids, h, inputs)
        return {"h": to_host(h)}, {name: to_host(a) for name, a in state.items()}


def load_stage(model_dir: str | Path, first: int, last: int) -> Stage:
    return Stage(model_dir, first, last)
