# Checklist for the first run with the Mac

Nothing below was run against real models or a real Mac yet. Do it in this order;
stop at the first red step.

## 0. Machine
- Close Kimi, 7-Zip and games. Power plan: not "PD Silent" (it halved the GEMM speed).
- Same network as the Mac; note whether the link is Wi-Fi, 2.5/1 GbE or a cable.

## 1. Tests (about a minute, GPU used lightly)
```
cd intel-helper/helper
..\.venv\Scripts\python -m pytest -q tests            # on the Intel GPU
set OMLX_HELPER_DEVICE=cpu && ..\.venv\Scripts\python -m pytest -q tests   # CPU control
```
WSL (real oMLX host code): see `xcheck/README.md`; `PYTHONPATH=~/omlx_src:~/helper pytest tests`.
Expect (CPU run, 2026-11-18): Windows 36 passed + 6 skipped (the skipped ones need the
oMLX source, i.e. WSL); WSL 42 passed, none skipped. On the GPU run the same, except a
failure there is news: tell me which test and the traceback. `tests/test_end_to_end.py`
is the closest thing to a Mac session: oMLX's own host code launches a real worker with
the real stage and replays the Mac-recorded fixtures over TCP.

## 2. Pair with the Mac (oMLX 0.8.0rc1)
1. Mac: Global Settings > Advanced > Enable Distributed Inference, restart oMLX.
2. This machine: `..\.venv\Scripts\python -m omlx_intel_helper.cli --state-dir %USERPROFILE%\.omlx-intel-helper serve --name my-intel-pc`
3. Cluster tab: Pair, type the code from the helper log.
4. Prefill Accelerator checks must show: control channel OK, model families
   `qwen3_5` and `qwen4_exp`, GPU memory. A "family" line that is red means the
   contract version disagrees: tell me the text.
5. If the helper is not found (mDNS blocked on Windows/Wi-Fi): start it with
   `--pair MAC_ADDRESS[:PORT]`.

## 3. Baseline and measurements (one number each)
1. Mac alone, Qwen3.8 27B, prompt of about 8K tokens: oMLX benchmark tok/s. => `--host-tps`
2. `python bench_stage.py <27B MLX folder> --layers N --chunk 2048 --tokens 8192 --host-tps <1.>`
   (N from its own "fit" line; it prints the estimated split speedup).
3. In oMLX model settings > Acceleration > Prefill Accelerator: choose the helper,
   run the same 8K prompt, read "via <helper>" speed.
4. Compare the answer text with and without the helper (same seed, greedy).

## 4. What to send me back
- the helper log (first 80 lines after loading a model and the first long prompt)
- the three speeds above, the link type, the power plan
- any "dropped layer" / "out of memory" lines

## Known weak spots to look at first if a number is bad
- Flash-Next MoE is a per-expert Python loop: expect it slow. Grouped GEMM next.
- GDN chunk scan is 64-token chunks with a triangular solve per layer; check it on
  the GPU profile before blaming the GEMMs.
- bf16 rounding differs slightly from Metal; a changed greedy continuation on some
  prompt is expected at this level (see README, "Known limits").
