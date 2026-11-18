"""Float64 'truth' for layer 0's projected qkv rows (the conv window), to tell
which side of the cross-check is closer: the XPU stage or the MLX CPU reference.

    python xcheck/truth_l0.py xcheck/tiny xcheck/stage_out.npz xcheck/ref_out.npz
"""

import json
import pathlib
import sys

import numpy as np
from safetensors import safe_open

model = pathlib.Path(sys.argv[1])
stage, ref = np.load(sys.argv[2]), np.load(sys.argv[3])
tokens = np.load(pathlib.Path(sys.argv[2]).with_suffix(".tokens.npy"))
cfg = json.loads((model / "config.json").read_text())
text, quant = cfg["text_config"], cfg["quantization"]
bits, group = quant["bits"], quant["group_size"]
root = "language_model.model."

f = safe_open(str(model / "model.safetensors"), "pt")


def dequant(prefix, rows=None):
    w = f.get_tensor(prefix + ".weight").view(__import__("torch").int32).numpy().astype(np.int64)
    s = f.get_tensor(prefix + ".scales").float().numpy().astype(np.float64)
    b = f.get_tensor(prefix + ".biases").float().numpy().astype(np.float64)
    if rows is not None:
        w, s, b = w[rows], s[rows], b[rows]
    per = 32 // bits
    codes = np.stack([(w >> (bits * j)) & (2**bits - 1) for j in range(per)], axis=-1)
    codes = codes.reshape(w.shape[0], -1).astype(np.float64)
    g = codes.reshape(w.shape[0], -1, group)
    return (g * s[..., None] + b[..., None]).reshape(w.shape[0], -1)


x = dequant(root + "embed_tokens", tokens)  # [n, hidden], exact
g = f.get_tensor(root + "layers.0.input_layernorm.weight").float().numpy().astype(np.float64)
eps = text["rms_norm_eps"]
normed = x / np.sqrt((x**2).mean(-1, keepdims=True) + eps) * g
W = dequant(root + "layers.0.linear_attn.in_proj_qkv")
qkv = normed @ W.T  # [n, conv]
truth = qkv[-3:]  # the conv window after the whole prompt

for label, arrays in (("stage (XPU)", stage), ("reference (MLX CPU)", ref)):
    got = arrays["200/0.0"][0].astype(np.float64)
    rel = np.linalg.norm(got - truth) / np.linalg.norm(truth)
    print(f"{label:22s} vs float64 truth: rel {rel:.2e}")
