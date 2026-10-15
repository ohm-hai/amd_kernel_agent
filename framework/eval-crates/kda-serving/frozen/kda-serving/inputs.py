"""Shared input generator for KDA decode/prefill benchmarks and verifiers.

Two gate modes:

- ``synthetic``: g = -|N(0, 1)|, beta = sigmoid(U(0, 1)). The original benchmark inputs.
- ``kimi``: the real KDA gate parameterization from FLA's layer
  (``fla/layers/kda.py``): g = -exp(A_log[h]) * softplus(x + dt_bias[h, k]), with
  A_log = log(U(1, 16)) and dt_bias = inverse-softplus(dt), dt = exp(U(log 1e-3, log 1e-1)),
  and beta = sigmoid(N(0, 1)). The gate pre-activation is x ~ N(0, GATE_X_SCALE), with the
  scale chosen so the p99 per-channel decay over 64 tokens is ~600 bits, the figure the
  KDA^2 blog reports for real Kimi-Linear prefills. Weight-trained distributions are not
  available here (no model download), so this is calibrated, not captured.
"""
import math

import torch

H_DEFAULT, K_DEFAULT, V_DEFAULT = 96, 128, 128
GATE_X_SCALE = 2.75  # calibrated 2026-10-09: p99 decay ~600 bits/64 tokens (see accuracy-bar.md)


def kimi_gate_params(H, K, seed=1234, device="cuda"):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    A_log = torch.log(torch.empty(H).uniform_(1, 16, generator=gen))
    dt = torch.exp(torch.rand(H * K, generator=gen) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    return A_log.to(device), dt_bias.view(H, K).to(device)


def kimi_gates(B, T, H, K, seed, device="cuda", x_scale=None):
    """Log-decay gates g <= 0 with FLA's KDA parameterization, fp32 [B, T, H, K]."""
    x_scale = GATE_X_SCALE if x_scale is None else x_scale
    A_log, dt_bias = kimi_gate_params(H, K, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.empty(B, T, H, K, device=device).normal_(0.0, x_scale, generator=gen)
    return -A_log.exp().view(1, 1, H, 1) * torch.nn.functional.softplus(x + dt_bias.view(1, 1, H, K))


def make_inputs(B, T, H=H_DEFAULT, K=K_DEFAULT, V=V_DEFAULT, seed=0, gates="synthetic", device="cuda"):
    """Return bf16 (q, k, v, g, beta) with q,k: [B,T,H,K], v: [B,T,H,V], g: [B,T,H,K], beta: [B,T,H]."""
    gen = torch.Generator(device=device).manual_seed(seed)
    normal = lambda *s: torch.empty(*s, device=device).normal_(generator=gen)
    q, k, v = normal(B, T, H, K), normal(B, T, H, K), normal(B, T, H, V)
    if gates == "synthetic":
        g = -normal(B, T, H, K).abs()
        beta = torch.rand(B, T, H, device=device, generator=gen).sigmoid()
    elif gates == "kimi":
        g = kimi_gates(B, T, H, K, seed=seed + 1, device=device)
        beta = normal(B, T, H).sigmoid()
    else:
        raise ValueError(f"unknown gates mode {gates!r}")
    return q.bfloat16(), k.bfloat16(), v.bfloat16(), g.bfloat16(), beta.bfloat16()


def decay_bits_per_64(g):
    """Per-channel cumulative decay over consecutive 64-token windows, in bits."""
    B, T, H, K = g.shape
    w = T // 64
    s = g[:, : w * 64].float().reshape(B, w, 64, H, K).sum(dim=2)
    return (-s / math.log(2)).flatten()


def calibrate(target_p99_bits=600.0, T=1024, H=H_DEFAULT, K=K_DEFAULT, device="cuda"):
    """Print decay percentiles per gate mode and return the x_scale whose p99 hits the target."""
    rows = []
    for scale in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        bits = decay_bits_per_64(kimi_gates(1, T, H, K, seed=7, device=device, x_scale=scale))
        q = torch.quantile(bits[torch.randperm(bits.numel(), device=bits.device)[:1_000_000]],
                           torch.tensor([0.5, 0.9, 0.99], device=bits.device))
        rows.append((scale, *q.tolist()))
    g_syn = -torch.randn(1, T, H, K, device=device).abs()
    qs = torch.quantile(decay_bits_per_64(g_syn)[:1_000_000], torch.tensor([0.5, 0.9, 0.99], device=device))
    best = min(rows, key=lambda r: abs(r[3] - target_p99_bits))
    return rows, qs.tolist(), best[0]


if __name__ == "__main__":
    rows, syn, best = calibrate()
    print("kimi gates: x_scale -> decay bits per 64 tokens (p50 / p90 / p99)")
    for scale, p50, p90, p99 in rows:
        print(f"  {scale:4.1f} -> {p50:8.1f} / {p90:8.1f} / {p99:8.1f}")
    print(f"synthetic -|N(0,1)|: {syn[0]:8.1f} / {syn[1]:8.1f} / {syn[2]:8.1f}")
    print(f"x_scale closest to p99 = 600 bits: {best}")
