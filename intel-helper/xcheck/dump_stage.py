"""Run the XPU stage on the tiny model and save hidden state and cache state.

    python xcheck/dump_stage.py xcheck/tiny xcheck/stage_out.npz 200 64,64,72
"""

import pathlib
import sys

import numpy as np
import torch

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "helper"))
sys.path.insert(0, str(root / "helper" / "tests"))
import test_qwen3_5_stage as T  # noqa: E402

from omlx_intel_helper.families.qwen3_5.stage import load_stage  # noqa: E402
from omlx_intel_helper.tensors import from_host, to_host  # noqa: E402

model, out, count = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
splits = [[count]] + [[int(x) for x in s.split(",")] for s in sys.argv[4:]]
tokens = T.TOKENS[:count]
np.save(pathlib.Path(out).with_suffix(".tokens.npy"), tokens)

stage = load_stage(model, 0, T.LAYERS)
result = {}
for sizes in splits:
    job = stage.new_job(T.LAYERS)
    start, hs, kv = 0, [], {}
    for size in sizes:
        o, state = stage.run(
            job, start=start, tokens=tokens[start : start + size], boundary=None,
            inputs={"rope": to_host(stage.frequencies)}, following=0, capture=None,
        )
        hs.append(from_host(o["h"]).float().cpu().numpy())
        for name, array in state.items():
            value = from_host(array).float().cpu().numpy()
            if name.endswith((".k", ".v")) and name in kv:
                value = np.concatenate([kv[name], value], axis=2)
            kv[name] = value
        start += size
    tag = "x".join(map(str, sizes))
    result[f"{tag}/h"] = np.concatenate(hs, axis=1)
    for name, value in kv.items():
        result[f"{tag}/{name}"] = value
np.savez(out, **result)
print("saved", out, len(result), "arrays")
