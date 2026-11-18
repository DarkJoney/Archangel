# omlx-intel-helper

Prefill Accelerator helper for oMLX (0.8.0rc1) on a machine with an Intel GPU
(Arc, Xe iGPU with XMX such as the Arc 140T of a Core Ultra 9 285H). Experimental.
A fork of `apps/omlx-cuda-helper` from oMLX (Apache-2.0): the Mac keeps the whole
model, the KV cache and the API; this helper holds the first layers of the model on
its GPU, runs them for each long prompt and sends the result to the Mac, which runs
the rest.

What is shared with the CUDA helper, unchanged: the wire protocol, the control
channel, pairing, discovery, the model store and the supervisor (a daemon with one
worker process per GPU). What is new: the whole compute side, on PyTorch XPU
instead of MLX CUDA.

## Status (2026-11-18)

| Part | State |
|---|---|
| Protocol, pairing, control channel | Ported; the tests drive oMLX's own host code (needs the oMLX source and MLX, i.e. Linux/WSL) |
| `qwen3_5` (Qwen3.5 / 3.6 / 3.8 dense, e.g. Qwen3.8 27B) | Ported (`families/qwen3_5`); matches the Mac-recorded contract fixture and real mlx-vlm in float32 |
| `qwen4_exp` (Qwen3.8 Flash-Next) | Ported (`families/qwen4_exp`) with plain-op reference paths; matches the Mac-recorded fixture on the tiny model |
| Speed | Measured once with `qwen3_5` 27B, 22 of 64 layers, over Wi-Fi: prefill +24 % at 8k, +47 % at 64k against the Mac alone, see `docs/RESULTS.md`. No fused kernels yet; see "Known limits" |

The first stage keeps the embedding; the output head, the final norm, the MTP
head and (Flash-Next) the PLE table stay on the Mac.

## Layout

```
omlx_intel_helper/
  cli.py supervisor.py worker.py control.py wire.py   daemon, stage worker processes, protocol
  pairing.py discovery.py state.py files.py           pairing, mDNS, identity, model copies
  gpus.py                                             Intel GPUs, memory, links (psutil + torch.xpu)
  tensors.py                                          torch helpers: safetensors load, wire <-> tensor, affine unpack
  checkpoint.py                                       safetensors headers and config, no GPU
  families/qwen3_5/  gdn.py layers.py stage.py        Gated DeltaNet scan, blocks, stage
  families/qwen4_exp/ ops.py attention.py layers.py stage.py
tests/    protocol, stage self-consistency and the Mac-recorded contract fixtures
xcheck/   (one level up) cross-check against real mlx-vlm on MLX CPU
bench_stage.py  stage speed on a real checkpoint
```

## Design notes

- A stage holds its linear layers unpacked to dense 16-bit (like the CUDA helper): no
  per-chunk dequantization; the number of layers it can hold is set by GPU memory.
  Expert banks (Flash-Next) stay packed and an expert is unpacked when it is used.
- Stages of one machine talk over loopback TCP with a per-session token, and
  `replaced` arrives on stdin: no `AF_UNIX` / `SIGUSR1`, so the same code runs on
  Windows and Linux.
- `OMLX_HELPER_DEVICE=cpu` runs stages on the CPU (tests only); default is `xpu`.
- `OMLX_HELPER_OS_RESERVE_GIB` (default 6): RAM an integrated GPU leaves to the OS.
- `OMLX_HELPER_SDPA=1`: qwen3_5 attention through torch's fused attention (16-bit
  scores) instead of the float32 score sheet.
- On some torch XPU builds advanced indexing (`a[ids]`) of a tensor with a
  2-element row crashes the process. The code uses `torch.index_select`
  (`ops.take`); keep it that way.

## Known limits

- No fused/custom kernels. The XMX path is whatever oneDNN gives torch (bf16/fp16
  GEMM); int8 DPAS (the fast path of other Intel engines) is not used because
  its numerics differ from the Mac's 16-bit prefill.
- Flash-Next MoE runs one expert at a time (a Python loop over the experts a chunk
  uses); a grouped GEMM is the first thing to write for speed.
- 16-bit reductions round once from float32; the CUDA helper imitates Metal's staged
  16-bit sums bit for bit. Differences are rounding noise, but they are not zero.
- A ranking of QSA blocks with tied scores takes the later blocks (stable sort),
  which matched the Mac's fixture; confirm on a real model.
- fp8 PLE rows, YaRN rotary scaling and images are not handled; the Mac host keeps
  such prompts local (`qwen3_5` adapter refuses YaRN).

## Run

```
pip install torch --index-url https://download.pytorch.org/whl/xpu
pip install numpy zeroconf psutil safetensors tokenizers
python -m omlx_intel_helper.cli --state-dir ~/.omlx-intel-helper serve --name arc-box
```

Pair it from the oMLX Cluster tab with the 6-digit code from the log. On Ubuntu see
`deploy/ubuntu/`.
