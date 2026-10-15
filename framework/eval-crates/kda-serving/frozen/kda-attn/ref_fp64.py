#!/usr/bin/env python3
"""Authoritative fp64 recurrent reference for KDAttn forward.

Faithful port of fla.ops.kda.naive.naive_recurrent_kda math to float64,
plus varlen (cu_seqlens) support. Slow by design — this is ground truth.

Semantics per step t (per head h):
    S <- S * exp(g_t)                       [per-key-channel decay]
    u_t = v_t - sum_k(k_t * S)              [delta: prediction error]
    S <- S + beta_t * k_t (outer) u_t       [delta-rule update]
    o_t = scale * q_t @ S
"""
import torch


def _recurrent_block(q, k, v, g, beta, scale, initial_state=None):
    """One legal-input sequence, fp64. Shapes: q,k [T,H,K]; v [T,HV,V]; g [T,HV,K]; beta [T,HV]."""
    T = q.shape[0]
    HV = v.shape[1]
    K, V = q.shape[-1], v.shape[-1]
    # KDAttn spec: Q and K are L2-normalized (fine-grained gating makes the
    # delta-rule contraction stable only on the unit sphere). Scale applies
    # AFTER normalization.
    q = q / q.norm(dim=-1, keepdim=True) * scale
    k = k / k.norm(dim=-1, keepdim=True)
    S = torch.zeros(HV, K, V, dtype=torch.float64, device=q.device)
    if initial_state is not None:
        S = S + initial_state.to(S)
    o = torch.empty(T, HV, V, dtype=torch.float64, device=q.device)
    for i in range(T):
        g_i = g[i].exp()                     # [HV, K]
        S = S * g_i[..., None]
        kv = torch.einsum("hk,hkv->hv", k[i], S)          # prediction [HV, V]
        u = v[i] - kv                                     # delta error
        S = S + torch.einsum("h,hk,hv->hkv", beta[i], k[i], u)
        o[i] = torch.einsum("hk,hkv->hv", q[i], S)
    return o, S


def kda_fp64_reference(q, k, v, g, beta, scale=None, cu_seqlens=None, output_final_state=True,
                       beta_sigmoid=False, initial_state=None):
    """fp64 ground truth.

    q,k: [B,T,H,K]; v: [B,T,HV,V]; g: [B,T,HV,K]; beta: [B,T,HV].
    beta_sigmoid=True: beta input is raw logits; sigmoid applied here in fp64
    (matches candidates using use_beta_sigmoid_in_kernel=True).
    cu_seqlens: int32/64 [N+1] — packed varlen layout.
    Returns (o [B,T,HV,V] fp64, S [B,HV,K,V] fp64 or None).
    """
    assert g.min() <= 0, "illegal input: log-decay gates must be <= 0"
    if scale is None:
        scale = q.shape[-1] ** -0.5
    q, k, v, g, beta = (x.to(torch.float64).contiguous() for x in (q, k, v, g, beta))
    if beta_sigmoid:
        beta = beta.sigmoid()
    if initial_state is not None and initial_state.dim() == 3:
        initial_state = initial_state.unsqueeze(0)  # [HV,K,V] -> [1,HV,K,V]
    B = q.shape[0]
    if cu_seqlens is None:
        outs, states = [], []
        for b in range(B):
            s0_b = None if initial_state is None else initial_state[b]
            o_b, s_b = _recurrent_block(q[b], k[b], v[b], g[b], beta[b], scale, s0_b)
            outs.append(o_b)
            states.append(s_b)
        o = torch.stack(outs)
        S = torch.stack(states) if output_final_state else None
        return o, S
    # varlen: packed [1, T_total, ...]
    cu = cu_seqlens.tolist()
    T_total = q.shape[1]
    outs = []
    states = []
    for s0, s1 in zip(cu[:-1], cu[1:]):
        o_s, s_s = _recurrent_block(
            q[0, s0:s1], k[0, s0:s1], v[0, s0:s1], g[0, s0:s1], beta[0, s0:s1], scale
        )
        outs.append(o_s)
        states.append(s_s)
    o = torch.cat(outs, dim=0).unsqueeze(0)
    S = torch.stack(states, dim=0) if output_final_state else None
    assert T_total == sum(s1 - s0 for s0, s1 in zip(cu[:-1], cu[1:]))
    return o, S


def rel_rmse(x, ref):
    """Relative RMSE (FlashKDA's own final-state metric)."""
    x, ref = x.to(torch.float64), ref.to(torch.float64)
    return ((x - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()


def max_abs_rel_err(x, ref, rel_floor_rms_frac=0.05, abs_floor=1e-3):
    """Strict element-wise errors with a scale-aware relative floor.

    max_abs: max |x-ref|. max_rel: max |x-ref| / max(|ref|, floor), where
    floor = max(abs_floor, rel_floor_rms_frac * rms(ref)) — prevents near-zero
    references (deep-decay states) from dominating, while still elementwise.
    """
    x, ref = x.to(torch.float64), ref.to(torch.float64)
    abs_err = (x - ref).abs()
    max_abs = abs_err.max().item()
    rms = ref.pow(2).mean().sqrt().item()
    floor = max(abs_floor, rel_floor_rms_frac * rms)
    max_rel = (abs_err / ref.abs().clamp_min(floor)).max().item()
    return max_abs, max_rel