"""Go/no-go benchmark for an Intel Arc prefill helper (PyTorch XPU).

Measures what the helper's hot path would cost on this GPU:
  1. bf16 / fp16 GEMM at layer-like shapes (TFLOPS)
  2. MLX-style affine 4-bit dequantisation (group size 64) to bf16 (GB/s)
  3. dequant + GEMM together, the way a layer would run per chunk

It prints an implied prompt speed for a dense 27B model (2 * 27e9 FLOP per
token, matmuls only) so the result can be compared with the Mac alone
(~405 tok/s) and with inferzero's int8 path on the 140T (~357 tok/s).

The shapes are representative, not read from a real Qwen3.8 config.

Run it when the GPU is idle: it shares the iGPU with a game.
    python bench_xpu.py            # quick, a few seconds of GPU time
    python bench_xpu.py --seconds 3 --chunks 2048 4096
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

GROUP = 64
FLOP_PER_TOKEN_27B = 2 * 27e9


def sync() -> None:
    torch.xpu.synchronize()


def timed(fn, seconds: float, min_iters: int = 3) -> tuple[float, int]:
    """Mean seconds per call after one warm-up call."""
    fn()
    sync()
    iters, start = 0, time.perf_counter()
    while True:
        fn()
        iters += 1
        sync()
        elapsed = time.perf_counter() - start
        if iters >= min_iters and elapsed >= seconds:
            return elapsed / iters, iters


def make_q4(n: int, k: int, dev, dtype):
    """Random weight in MLX affine layout: packed uint32 [n, k/8] (as int32),
    scales and biases [n, k/GROUP]."""
    packed = torch.randint(
        -(2**31), 2**31 - 1, (n, k // 8), dtype=torch.int32, device=dev
    )
    scales = (torch.rand(n, k // GROUP, device=dev) * 0.02).to(dtype)
    biases = (torch.rand(n, k // GROUP, device=dev) * -0.1).to(dtype)
    return packed, scales, biases


def dequant(packed, scales, biases, dtype):
    n, kp = packed.shape
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int32)
    q = (packed.unsqueeze(-1) >> shifts) & 0xF  # [n, k/8, 8], low nibble first
    q = q.reshape(n, kp * 8).to(dtype)
    g = q.reshape(n, -1, GROUP)
    w = g * scales.unsqueeze(-1) + biases.unsqueeze(-1)
    return w.reshape(n, -1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=1.5, help="per measurement")
    ap.add_argument("--chunks", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument(
        "--max-gib", type=float, default=4.0, help="stop if a case needs more"
    )
    args = ap.parse_args()

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        print("torch.xpu is not available: no XPU device or driver/runtime missing")
        print("torch", torch.__version__)
        return 1
    dev = torch.device("xpu")
    props = torch.xpu.get_device_properties(0)
    print(f"torch {torch.__version__}")
    print(f"device: {props.name}, memory visible: {props.total_memory / 2**30:.1f} GiB")
    print(f"sub-groups: {getattr(props, 'sub_group_sizes', 'n/a')}")
    print()

    # Representative dense-layer matrices (hidden 5120, FFN 17408).
    shapes = {
        "ffn up/gate": (5120, 17408),
        "ffn down": (17408, 5120),
        "attn proj": (5120, 5120),
    }
    for dtype in (torch.bfloat16, torch.float16):
        print(f"== GEMM {dtype} ==")
        for m in args.chunks:
            for name, (k, n) in shapes.items():
                need = (m * k + k * n + m * n) * 2 / 2**30
                if need > args.max_gib:
                    print(f"  M={m} {name}: skipped ({need:.1f} GiB)")
                    continue
                a = torch.randn(m, k, device=dev, dtype=dtype)
                w = torch.randn(n, k, device=dev, dtype=dtype)
                sec, _ = timed(lambda: torch.nn.functional.linear(a, w), args.seconds)
                tflops = 2 * m * k * n / sec / 1e12
                print(
                    f"  M={m:5d} {name:12s} K={k:5d} N={n:5d}: "
                    f"{tflops:6.2f} TFLOPS  (~{tflops * 1e12 / FLOP_PER_TOKEN_27B:5.0f} tok/s on 27B)"
                )
                del a, w
        print()

    dtype = torch.bfloat16
    print("== affine 4-bit dequant (group 64) -> bf16 ==")
    k, n = 5120, 17408
    packed, scales, biases = make_q4(n, k, dev, dtype)
    sec, _ = timed(lambda: dequant(packed, scales, biases, dtype), args.seconds)
    moved = packed.numel() * 4 + n * k * 2  # read packed, write bf16
    print(f"  one {n}x{k} matrix: {sec * 1e3:7.2f} ms  ({moved / sec / 1e9:6.1f} GB/s)")
    print()

    print("== dequant + GEMM per chunk (JIT weights, bf16) ==")
    for m in args.chunks:
        need = (m * k + n * k * 2 + m * n) * 2 / 2**30
        if need > args.max_gib:
            print(f"  M={m}: skipped ({need:.1f} GiB)")
            continue
        a = torch.randn(m, k, device=dev, dtype=dtype)

        def both():
            w = dequant(packed, scales, biases, dtype)
            return torch.nn.functional.linear(a, w)

        def gemm_only():
            return torch.nn.functional.linear(a, w_full)

        w_full = dequant(packed, scales, biases, dtype)
        s_both, _ = timed(both, args.seconds)
        s_gemm, _ = timed(gemm_only, args.seconds)
        t_both = 2 * m * k * n / s_both / 1e12
        print(
            f"  M={m:5d}: gemm {s_gemm * 1e3:7.1f} ms, with dequant {s_both * 1e3:7.1f} ms "
            f"(+{(s_both / s_gemm - 1) * 100:4.0f}%), {t_both:5.2f} TFLOPS "
            f"~{t_both * 1e12 / FLOP_PER_TOKEN_27B:4.0f} tok/s"
        )
        del a, w_full
    print()
    print("Reference: Mac alone ~405 tok/s (27B, 32K); inferzero int8 on 140T ~357 tok/s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
