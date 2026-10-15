#!/usr/bin/env python3
"""Hardened decode verifier for KDA candidates (thresholds: ../accuracy-bar.md).

The candidate steps one token at a time and carries its own state; every result is
compared with the fp64 token-by-token recurrence on the same bf16 inputs.

    HIP_VISIBLE_DEVICES=<free gpu> python3 verifier/verify_decode.py --candidate runs/<name> [--quick]

Candidate module contract (runs/<name>/candidate.py):
    get_forward() -> fwd(q, k, v, g, beta, initial_state=..., output_final_state=True)
        T=1 inputs: q,k,g [B,1,H,K], v [B,1,H,V], beta [B,1,H]  (bf16)
        returns o [B,1,H,V], or (o, S); otherwise the new state is fwd.last_final_state
    STATE_DTYPE (optional, default torch.float32): dtype the candidate stores state in
    INPLACE (optional, default False): candidate may overwrite initial_state
Writes runs/<name>/verify_decode_report.json and exits 0 only if every gate passes.
"""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(TASK.parent / "kda-attn"))
sys.path.insert(0, str(TASK.parent / "tools"))
import inputs as inp  # noqa: E402
from env_stamp import env_stamp, require_free_gpu  # noqa: E402
from ref_fp64 import kda_fp64_reference, rel_rmse  # noqa: E402

DEV = "cuda"
K = V = 128
# Pre-registered bar (accuracy-bar.md, 2026-10-09). Do not edit without user approval.
TOL_O_RRMSE = 8e-3
TOL_O_MAX_ABS = 5e-2
TOL_STATE_RRMSE = 1e-2
SEEDS = (11, 427, 90210)


def load_candidate(path):
    spec = importlib.util.spec_from_file_location("candidate", Path(path) / "candidate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def step_candidate(fwd, q, k, v, g, beta, S0, state_dtype, inplace):
    """Run T single-token steps. Returns (o [B,T,H,V] fp32, final state, purity_ok)."""
    S = S0.to(state_dtype).clone()
    guard = S.clone()
    outs, pure = [], True
    for t in range(q.shape[1]):
        sl = slice(t, t + 1)
        res = fwd(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl],
                  initial_state=S, output_final_state=True)
        o, S_new = res if isinstance(res, tuple) else (res, fwd.last_final_state)
        if t == 0 and not inplace and not torch.equal(S, guard):
            pure = False
        outs.append(o.float())
        S = S_new
    return torch.cat(outs, dim=1), S, pure


def windows(T):
    if T <= 192:
        return [(0, T)]
    mid = (T // 2) // 64 * 64
    return [(0, 64), (mid, mid + 64), (T - 64, T)]


def check(name, fwd, mod, q, k, v, g, beta, S0=None):
    B, T, H, _ = q.shape
    if S0 is None:
        S0 = torch.zeros(B, H, K, V, dtype=torch.float64, device=DEV)
    state_dtype = getattr(mod, "STATE_DTYPE", torch.float32)
    inplace = getattr(mod, "INPLACE", False)
    t0 = time.perf_counter()
    o, S, pure = step_candidate(fwd, q, k, v, g, beta, S0, state_dtype, inplace)
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    with torch.no_grad():
        o_ref, S_ref = kda_fp64_reference(q, k, v, g, beta, initial_state=S0)
    result = {"gate": name, "B": B, "T": T, "H": H, "seconds": round(secs, 2)}
    if not (torch.isfinite(o).all() and torch.isfinite(S.float()).all()):
        result.update(passed=False, detail="non-finite output or state")
        return result
    o_rrmse = max(rel_rmse(o[:, a:b], o_ref[:, a:b]) for a, b in windows(T))
    o_abs = (o.double() - o_ref).abs().max().item()
    s_rrmse = rel_rmse(S.double(), S_ref)
    passed = pure and o_rrmse <= TOL_O_RRMSE and o_abs <= TOL_O_MAX_ABS and s_rrmse <= TOL_STATE_RRMSE
    result.update(passed=bool(passed), o_rrmse_worst_window=o_rrmse, o_max_abs=o_abs,
                  state_rrmse=s_rrmse, input_state_untouched=pure)
    return result


def warm_state(B, H, L, seed, gates="kimi"):
    """fp64 state after prefilling L tokens (computed by the reference, not the candidate)."""
    q, k, v, g, beta = inp.make_inputs(B, L, H=H, seed=seed, gates=gates)
    with torch.no_grad():
        _, S = kda_fp64_reference(q, k, v, g, beta)
    return S


def run_gates(fwd, mod, quick):
    torch.manual_seed(1234)  # stress tensors below are reproducible run to run
    res = []

    def add(r):
        res.append(r)
        tag = "PASS" if r["passed"] else "FAIL"
        extra = r.get("detail") or (f"o_rRMSE {r['o_rrmse_worst_window']:.3g}  "
                                    f"o_abs {r['o_max_abs']:.3g}  state_rRMSE {r['state_rrmse']:.3g}")
        print(f"[{tag}] {r['gate']:28s} B={r['B']} H={r['H']} T={r['T']:5d}  {extra}", flush=True)

    # D1: salts x gate modes, from zero state
    for gates in ("synthetic", "kimi"):
        for seed in SEEDS[: 1 if quick else 3]:
            add(check(f"D1_{gates}_s{seed}", fwd, mod, *inp.make_inputs(2, 256, H=8, seed=seed, gates=gates)))

    # D2: warm state from a 2048-token fp64 prefill, then 64 decode steps
    S_w = warm_state(2, 8, 2048, seed=5)
    add(check("D2_warm_state_2048", fwd, mod, *inp.make_inputs(2, 64, H=8, seed=6, gates="kimi"), S0=S_w))

    # D3: stress probes
    q, k, v, g, beta = inp.make_inputs(2, 128, H=8, seed=21, gates="kimi")
    per_tok = 600.0 / 64.0 * math.log(2)  # ~600 bits per 64 tokens, uniform
    g_deep = (-per_tok * torch.empty_like(g, dtype=torch.float32).uniform_(0.5, 1.5)).bfloat16()
    add(check("D3_deep_decay", fwd, mod, q, k, v, g_deep, beta))
    q2, k2, v2, g2, b2 = inp.make_inputs(2, 256, H=8, seed=22, gates="kimi")
    g_unity = (-torch.rand_like(g2, dtype=torch.float32) * 1e-3).bfloat16()
    add(check("D3_near_unity_gates", fwd, mod, q2, k2, v2, g_unity, b2))
    b_ext = torch.where(torch.rand_like(beta, dtype=torch.float32) < 0.5, 1e-3, 1 - 1e-3).bfloat16()
    add(check("D3_beta_extremes", fwd, mod, q, k, v, g, b_ext))
    add(check("D3_zero_v_from_warm", fwd, mod, q[:, :64], k[:, :64], torch.zeros_like(v[:, :64]),
              g[:, :64], beta[:, :64], S0=S_w))
    add(check("D3_large_v_x32", fwd, mod, q, k, (v.float() * 32).bfloat16(), g, beta))

    # D4: long-run drift, where reduced-precision state fails
    if not quick:
        add(check("D4_drift_8192_kimi", fwd, mod, *inp.make_inputs(1, 8192, H=8, seed=31, gates="kimi")))

    # D5: shapes (odd batch, single head, Kimi K3 head count)
    add(check("D5_B3_H1", fwd, mod, *inp.make_inputs(3, 32, H=1, seed=41, gates="kimi")))
    add(check("D5_B2_H96_kimi_k3", fwd, mod, *inp.make_inputs(2, 16, H=96, seed=42, gates="kimi")))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--quick", action="store_true", help="1 salt per gate mode, skip the 8192-step drift gate")
    ap.add_argument("--allow-busy-gpu", action="store_true")
    args = ap.parse_args()
    if not args.allow_busy_gpu:
        require_free_gpu()
    mod = load_candidate(args.candidate)
    fwd = mod.get_forward()
    print(f"verifying {args.candidate} (state dtype {getattr(mod, 'STATE_DTYPE', torch.float32)})")
    t0 = time.perf_counter()
    res = run_gates(fwd, mod, args.quick)
    passed = all(r["passed"] for r in res)
    report = {
        "candidate": args.candidate, "passed": passed, "quick": args.quick,
        "bar": {"o_rrmse": TOL_O_RRMSE, "o_max_abs": TOL_O_MAX_ABS, "state_rrmse": TOL_STATE_RRMSE,
                "source": "accuracy-bar.md (pre-registered 2026-10-09)"},
        "n_gates": len(res), "n_failed": sum(not r["passed"] for r in res),
        "seconds": round(time.perf_counter() - t0, 1), "env": env_stamp(), "results": res,
    }
    out = Path(args.candidate) / ("verify_decode_report_quick.json" if args.quick else "verify_decode_report.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\n{'ACCEPTED' if passed else 'REJECTED'}: {report['n_gates'] - report['n_failed']}/{report['n_gates']} "
          f"gates passed in {report['seconds']}s -> {out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
