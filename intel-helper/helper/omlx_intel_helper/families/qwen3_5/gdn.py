# SPDX-License-Identifier: Apache-2.0
"""Gated delta rule for prefill (PyTorch): a per-token reference and the
chunked parallel scan.

Ports of mlx-vlm's ``gated_delta_ops`` and ``gated_delta_chunked`` (MIT,
Copyright (c) 2025 Prince Canuma; vendored from mlx-lm). Scalar gating only.

Shapes: q, k [B, T, Hk, Dk]; v [B, T, Hv, Dv]; g, beta [B, T, Hv];
state [B, Hv, Dv, Dk] in float32. Returns y [B, T, Hv, Dv] in v's dtype and
the new state (float32).
"""

from __future__ import annotations

import torch

CHUNK = 64


def sequential(q, k, v, g, beta, state):
    """One step per token: the definition the chunked scan must equal."""
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    in_dtype = v.dtype
    if state is None:
        state = torch.zeros(B, Hv, Dv, Dk, dtype=torch.float32, device=q.device)
    if (rf := Hv // Hk) > 1:
        q = q.repeat_interleave(rf, dim=-2)
        k = k.repeat_interleave(rf, dim=-2)
    q, k, v = q.float(), k.float(), v.float()
    g, beta = g.float(), beta.float()
    ys = []
    for t in range(T):
        state = state * g[:, t, :, None, None]
        kv_mem = (state * k[:, t, :, None, :]).sum(-1)
        delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
        state = state + k[:, t, :, None, :] * delta[..., None]
        ys.append((state * q[:, t, :, None, :]).sum(-1))
    return torch.stack(ys, dim=1).to(in_dtype), state


_solve_ok: bool | None = None


def _invert_unit_lower(T: torch.Tensor, C: int) -> torch.Tensor:
    """Inverse of a batched unit lower-triangular matrix (forward
    substitution; stable where a Neumann series overflows)."""
    global _solve_ok
    eye = torch.eye(C, dtype=T.dtype, device=T.device)
    if _solve_ok is not False:
        try:
            out = torch.linalg.solve_triangular(
                T, eye.expand_as(T), upper=False, unitriangular=True
            )
            _solve_ok = True
            return out
        except (RuntimeError, NotImplementedError):
            _solve_ok = False
    rows = [eye.expand_as(T)[..., 0:1, :]]
    for i in range(1, C):
        Tij = T[..., i : i + 1, :i]
        Xj = torch.cat(rows, dim=-2)
        rows.append(eye.expand_as(T)[..., i : i + 1, :] - Tij @ Xj)
    return torch.cat(rows, dim=-2)


def chunked(q, k, v, g, beta, state, C: int = CHUNK):
    """The chunked parallel scan: intra-chunk matmuls, a C-sized triangular
    inverse and a T/C-step scan of the state."""
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    in_dtype = v.dtype
    dev = q.device
    if (rf := Hv // Hk) > 1:
        q = q.repeat_interleave(rf, dim=-2)
        k = k.repeat_interleave(rf, dim=-2)
    g, beta = g.float(), beta.float()
    pad = (C - T % C) % C
    if pad:
        z = lambda *s, dt: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731
        q = torch.cat([q, z(B, pad, Hv, Dk, dt=q.dtype)], dim=1)
        k = torch.cat([k, z(B, pad, Hv, Dk, dt=k.dtype)], dim=1)
        v = torch.cat([v, z(B, pad, Hv, Dv, dt=v.dtype)], dim=1)
        g = torch.cat([g, torch.ones(B, pad, Hv, dtype=g.dtype, device=dev)], dim=1)
        beta = torch.cat([beta, z(B, pad, Hv, dt=beta.dtype)], dim=1)
    Tp = T + pad
    nC = Tp // C

    def rc(x, D):
        return x.reshape(B, nC, C, Hv, D).permute(0, 3, 1, 2, 4).float()

    q, k, v = rc(q, Dk), rc(k, Dk), rc(v, Dv)  # [B, Hv, nC, C, D]
    g = g.reshape(B, nC, C, Hv).permute(0, 3, 1, 2)  # [B, Hv, nC, C]
    beta = beta.reshape(B, nC, C, Hv).permute(0, 3, 1, 2)

    # g can underflow to exactly 0 in float32; log(0) would poison the ratios.
    lcg = torch.cumsum(torch.log(g.clamp(1e-6, 1.0)), dim=-1)
    cumg = torch.exp(lcg)
    ones = torch.ones(C, C, dtype=torch.float32, device=dev)
    lower_incl, strict_lower = torch.tril(ones, 0), torch.tril(ones, -1)
    # Mask the exponent to the lower triangle before exp: the upper one
    # would overflow and inf * 0 is nan.
    diff = torch.where(
        lower_incl > 0, lcg[..., :, None] - lcg[..., None, :], torch.tensor(-1e30, device=dev)
    )
    dr = torch.exp(diff)

    KK = k @ k.transpose(-1, -2)
    A = beta[..., :, None] * dr * KK * strict_lower
    Tinv = _invert_unit_lower(torch.eye(C, dtype=torch.float32, device=dev) + A, C)

    kbeta = (beta * cumg)[..., :, None] * k
    U0 = Tinv @ (beta[..., :, None] * v)  # [B, Hv, nC, C, Dv]
    Kt = Tinv @ kbeta  # [B, Hv, nC, C, Dk]
    M = dr * (q @ k.transpose(-1, -2)) * lower_incl
    Qeff = cumg[..., :, None] * q - M @ Kt
    MU0 = M @ U0
    cumg_last = cumg[..., -1]
    ratio_last = torch.exp(lcg[..., -1, None] - lcg)  # cumg_last / cumg_i

    S = (
        state.float()
        if state is not None
        else torch.zeros(B, Hv, Dv, Dk, dtype=torch.float32, device=dev)
    )
    ys = []
    for c in range(nC):
        St = S.transpose(-1, -2)  # [B, Hv, Dk, Dv]
        ys.append(MU0[:, :, c] + Qeff[:, :, c] @ St)  # [B, Hv, C, Dv]
        Uc = U0[:, :, c] - Kt[:, :, c] @ St
        Uscaled = ratio_last[:, :, c][..., None] * Uc
        S = (
            cumg_last[:, :, c][..., None, None] * S
            + Uscaled.transpose(-1, -2) @ k[:, :, c]
        )
    Y = torch.stack(ys, dim=2).permute(0, 2, 3, 1, 4).reshape(B, Tp, Hv, Dv)[:, :T]
    return Y.to(in_dtype), S


def update(q, k, v, g, beta, state):
    """Prefill update: the chunked scan for more than one token."""
    if state is None:
        B, _, _, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = torch.zeros(B, Hv, Dv, Dk, dtype=torch.float32, device=q.device)
    if q.shape[1] > 1:
        return chunked(q, k, v, g, beta, state)
    return sequential(q, k, v, g, beta, state)
