"""Write the tiny random MLX-layout Qwen3.5 checkpoint used by the cross-check.

    python xcheck/make_tiny.py xcheck/tiny [float32|bfloat16]
"""

import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root / "helper"))
sys.path.insert(0, str(_root / "helper" / "tests"))
import test_qwen3_5_stage as T  # noqa: E402

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
import torch
T.make_model(out, getattr(torch, sys.argv[2]) if len(sys.argv) > 2 else torch.bfloat16)
print("wrote", out)
