import math, time, torch
import torch.nn.functional as F

dev = torch.device("xpu")
H, KV, D, R = 24, 4, 256, 2048
G = H // KV


def sync():
    torch.xpu.synchronize()


def sheet(q, k, v, past, step=None):
    total = k.shape[2]
    grouped = q.reshape(1, KV, G, R, D)
    kt = k.float().transpose(-1, -2).contiguous()
    vf = v.float()
    step = step or max(1, (1 << 30) // (4 * H * total))
    cols = torch.arange(total, device=dev)
    outs = []
    for s in range(0, R, step):
        e = min(s + step, R)
        n = e - s
        qq = grouped[..., s:e, :].float() / math.sqrt(D)
        sc = qq.reshape(1, KV, G * n, D) @ kt
        vis = (past + torch.arange(s, e, device=dev))[:, None] >= cols
        sc = sc.reshape(1, KV, G, n, total).masked_fill(~vis, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        o = p.reshape(1, KV, G * n, total) @ vf
        outs.append(o.reshape(1, H, n, D))
    return torch.cat(outs, 2).to(q.dtype)


def tiled_sdpa(q, k, v, past, block=512, gqa=True):
    total = k.shape[2]
    cols = torch.arange(total, device=dev)[None]
    outs = []
    if not gqa:
        k = k.repeat_interleave(G, 1)
        v = v.repeat_interleave(G, 1)
    for s in range(0, R, block):
        e = min(s + block, R)
        vis = (past + torch.arange(s, e, device=dev))[:, None] >= cols
        kw = {"enable_gqa": True} if gqa else {}
        outs.append(F.scaled_dot_product_attention(q[:, :, s:e], k, v, attn_mask=vis, scale=1 / math.sqrt(D), **kw))
    return torch.cat(outs, 2)


def timeit(f, n=3):
    f(); sync()
    t = time.perf_counter()
    for _ in range(n):
        f()
    sync()
    return (time.perf_counter() - t) / n


for total in (2048, 16384, 32768):
    past = total - R
    q = torch.randn(1, H, R, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, KV, total, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, KV, total, D, device=dev, dtype=torch.bfloat16)
    ref = sheet(q, k, v, past)
    print(f"total {total}")
    print("  sheet fp32      %.3fs" % timeit(lambda: sheet(q, k, v, past)))
    for name, fn in (
        ("sdpa tiled gqa", lambda: tiled_sdpa(q, k, v, past)),
        ("sdpa tiled rep", lambda: tiled_sdpa(q, k, v, past, gqa=False)),
        ("sdpa tiled 256", lambda: tiled_sdpa(q, k, v, past, block=256)),
    ):
        try:
            t = timeit(fn)
            err = (fn().float() - ref.float()).abs().max().item()
            print(f"  {name:15s} {t:.3f}s  maxerr {err:.3g}")
        except Exception as e:
            print(f"  {name:15s} FAIL {str(e)[:90]}")
    del q, k, v, ref
    torch.xpu.empty_cache()
