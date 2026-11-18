"""Reference: the same tiny checkpoint through mlx-vlm's own Qwen3.5 language
model, with the layer loop the oMLX adapter uses (adapters/qwen3_5.py).

Runs on MLX CPU (Linux). Saves hidden state before the final norm and the cache
state in the same names as dump_stage.py.

    python ref_mlx.py tiny ref_out.npz 200 64,64,72
"""

import json
import pathlib
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.cache import ArraysCache, KVCache
from mlx_vlm.models.qwen3_5 import language
from mlx_vlm.models.qwen3_5.config import TextConfig

model, out, count = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
splits = [[count]] + [[int(x) for x in s.split(",")] for s in sys.argv[4:]]
tokens = np.load(pathlib.Path(sys.argv[2]).with_suffix(".tokens.npy").as_posix().replace("ref_out", "stage_out"))[:count]

cfg = json.loads((model / "config.json").read_text())
text = dict(cfg["text_config"])
text.update(model_type="qwen3_5", max_position_embeddings=4096)
rope = dict(text["rope_parameters"])
rope.update(type="default", mrope_section=[2, 1, 1])
text["rope_parameters"] = rope
args = TextConfig(**text)
lm = language.LanguageModel(args)

weights = mx.load(str(model / "model.safetensors"))
weights = {k.removeprefix("language_model."): v for k, v in weights.items()}
quant = cfg["quantization"]
nn.quantize(
    lm,
    group_size=quant["group_size"],
    bits=quant["bits"],
    class_predicate=lambda path, m: f"{path}.scales" in weights,
)
lm.load_weights(list(weights.items()), strict=False)
mx.eval(lm.parameters())
layers = lm.model.layers


def run_layers(h, cache):
    pairs = list(zip(layers, cache))
    attention = next(((layer, c) for layer, c in pairs if not layer.is_linear), None)
    linear = next((c for layer, c in pairs if layer.is_linear), None)
    fa_mask = ssm_mask = positions = embeddings = None
    if linear is not None:
        ssm_mask = language._create_qwen3_5_ssm_mask(h, linear)
    if attention is not None:
        layer, item = attention
        fa_mask = language._create_qwen3_5_attention_mask(h, item)
        positions = mx.arange(item.offset, item.offset + h.shape[1], dtype=mx.int32)
        positions = mx.broadcast_to(positions[None, None], (3, *h.shape[:2]))
        rotary = layer.self_attn.rotary_emb
        if not getattr(rotary, "fused_apply", False):
            embeddings = rotary(h, positions)
    for layer, c in pairs:
        h = layer(
            h,
            mask=ssm_mask if layer.is_linear else fa_mask,
            cache=c,
            position_ids=positions,
            position_embeddings=embeddings,
        )
    return h


result = {}
for sizes in splits:
    cache = [ArraysCache(size=2) if l.is_linear else KVCache() for l in layers]
    start, hs, before = 0, [], [0] * len(layers)
    kv = {}
    for size in sizes:
        ids = mx.array(tokens[start : start + size].astype(np.int32))[None]
        h = run_layers(lm.model.embed_tokens(ids), cache)
        mx.eval(h)
        hs.append(np.array(h.astype(mx.float32)))
        start += size
    for i, c in enumerate(cache):
        if isinstance(c, ArraysCache):
            for s, v in enumerate(c.cache):
                kv[f"{i}.{s}"] = np.array(v.astype(mx.float32))
        else:
            kv[f"{i}.k"] = np.array(c.keys[..., : c.offset, :].astype(mx.float32))
            kv[f"{i}.v"] = np.array(c.values[..., : c.offset, :].astype(mx.float32))
    tag = "x".join(map(str, sizes))
    result[f"{tag}/h"] = np.concatenate(hs, axis=1)
    for name, value in kv.items():
        result[f"{tag}/{name}"] = value
np.savez(out, **result)
print("saved", out, len(result), "arrays")
