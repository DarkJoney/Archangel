# Cross-check against the real MLX model

Same tiny random Qwen3.5 checkpoint, same tokens, two implementations:
the XPU stage (this repo, Windows or Linux with an Intel GPU) and mlx-vlm's own
`qwen3_5` language model on MLX CPU (WSL / Linux), with the layer loop of the
oMLX adapter. Compares the hidden state and every layer's cache state.

```
# 1. checkpoint (float32 checks the logic; bfloat16 shows rounding noise)
python xcheck/make_tiny.py xcheck/tiny32 float32
# 2. XPU stage (Windows or Linux with torch+xpu)
python xcheck/dump_stage.py xcheck/tiny32 xcheck/stage_out.npz 200 64,64,72
# 3. reference (WSL / Linux; uv venv with: mlx[cpu] mlx-lm mlx-vlm numpy safetensors)
cd xcheck && python ref_mlx.py tiny32 ref_out.npz 200 64,64,72
# 4. compare
python xcheck/compare.py xcheck/stage_out.npz xcheck/ref_out.npz
```

Result (2026-11-17, Arc 140T vs MLX 0.32.4 CPU, float32): worst relative error
5e-6, whole prompt and chunked alike. In bfloat16 the MLX CPU reference is the
noisier side (layer 0 qkv: 1.4e-2 from a float64 truth, the XPU stage 3e-3;
`truth_l0.py`), so a bfloat16 comparison is not a tight check.

Not covered: the Mac's Metal kernels (the numbers the Mac host really produces),
YaRN, images, MTP priming. Compare on the Mac itself when it is available.
