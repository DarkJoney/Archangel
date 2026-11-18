"""Compare stage_out.npz (XPU) with ref_out.npz (MLX): relative error per array."""

import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
worst = 0.0
for name in sorted(a.files):
    if name not in b.files:
        print("MISSING in reference:", name)
        continue
    x, y = a[name].astype(np.float64), b[name].astype(np.float64)
    if x.shape != y.shape:
        print(f"{name}: SHAPE {x.shape} vs {y.shape}")
        continue
    rel = np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30)
    mx_abs = np.abs(x - y).max()
    worst = max(worst, rel)
    print(f"{name:14s} rel {rel:9.2e}  max|d| {mx_abs:9.2e}")
print("worst relative error:", f"{worst:.2e}")
