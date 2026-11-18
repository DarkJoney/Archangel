"""Speed of the qwen3_5 helper stage on a real MLX checkpoint (layers [0, K)).

Prefill throughput of the helper side alone, in prompt tokens per second, for the
layers it would hold. With ``--host-tps`` (the Mac's own prefill speed for the
WHOLE model, from oMLX's benchmark) it also estimates the split speedup.

    python bench_stage.py D:\\models\\Qwen3.8-27B-oQ4e --layers 24 --chunk 2048 --tokens 8192 --host-tps 120

Run it on an idle GPU (it shares the iGPU with everything else). Each chunk
runs through all K layers like a real job: chunks of one prompt keep their
state, and the numbers of the first chunk (kernel compilation) are reported
apart.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from omlx_intel_helper import checkpoint, gpus
from omlx_intel_helper.families import qwen3_5
from omlx_intel_helper.families.qwen3_5.stage import load_stage
from omlx_intel_helper.tensors import to_host


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", type=Path, help="MLX checkpoint folder")
    ap.add_argument("--layers", type=int, default=0, help="layers to hold (0: as many as fit)")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--host-tps", type=float, default=0.0, help="Mac prefill tok/s, whole model")
    ap.add_argument("--reserve-gib", type=float, default=3.0, help="memory kept for activations")
    args = ap.parse_args()

    if not torch.xpu.is_available():
        print("no XPU")
        return 1
    shared, per_layer = qwen3_5.plan(args.model)
    total = len(per_layer)
    gpu = gpus.gpus()[0]
    budget = gpus.memory_limit(gpu) - int(args.reserve_gib * 2**30)
    fit, used = 0, shared
    for size in per_layer:
        if used + size > budget:
            break
        used, fit = used + size, fit + 1
    layers = args.layers or fit
    print(f"{gpu['name']}: budget {budget / 2**30:.1f} GiB, model has {total} layers "
          f"({shared / 2**30:.2f} GiB shared + {sum(per_layer) / 2**30:.1f} GiB of layers, dense)")
    print(f"holding {layers} of {total} layers ({fit} fit)")
    if layers > fit:
        print("warning: more layers than the budget; may run out of memory")

    began = time.perf_counter()
    stage = load_stage(args.model, 0, layers)
    torch.xpu.synchronize()
    print(f"loaded in {time.perf_counter() - began:.1f} s, "
          f"{stage.memory_bytes() / 2**30:.1f} GiB allocated")

    vocab = stage.vocab
    tokens = (np.arange(args.tokens) * 2654435761 % (vocab - 3) + 3).astype(np.int32)
    sizes = [args.chunk] * (args.tokens // args.chunk)
    if args.tokens % args.chunk:
        sizes.append(args.tokens % args.chunk)
    job = stage.new_job(layers)
    inputs = {"rope": to_host(stage.frequencies)}
    stage.peak_bytes()
    start, times = 0, []
    for index, size in enumerate(sizes):
        t0 = time.perf_counter()
        stage.run(job, start=start, tokens=tokens[start : start + size], boundary=None,
                  inputs=inputs, following=0, capture=None)
        torch.xpu.synchronize()
        times.append((size, time.perf_counter() - t0))
        print(f"  chunk {index}: {size} tokens in {times[-1][1]:.2f} s = "
              f"{size / times[-1][1]:.0f} tok/s")
        start += size
    steady = times[1:] or times
    rate = sum(s for s, _ in steady) / sum(t for _, t in steady)
    print(f"steady {rate:.0f} tok/s for {layers} layers; peak "
          f"{stage.peak_bytes() / 2**30:.1f} GiB")
    # All layers take about the same time (a GDN layer and an attention layer
    # differ; this is the mean), so the whole model would run at about:
    whole = rate * layers / total
    print(f"this GPU alone, whole model: ~{whole:.0f} tok/s")
    if args.host_tps:
        # Helper, link and Mac work on different chunks at once: the slowest
        # share sets the pace. The helper's layers take share f of the work.
        f = layers / total
        helper_time = 1.0 / rate  # seconds per token for the helper's layers
        host_time = (1 - f) / args.host_tps  # the Mac's share of its own time
        split = 1.0 / max(helper_time, host_time)
        print(f"with the Mac ({args.host_tps:.0f} tok/s alone): ~{split:.0f} tok/s "
              f"({split / args.host_tps:.2f}x), before link and chunk-start costs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
