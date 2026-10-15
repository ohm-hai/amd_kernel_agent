#!/usr/bin/env python3
"""Benchmark harness for KDAttn candidates on MI355X.

Mirrors the blog's evaluation set: fixed-length 8192 + varlen mixed/uniform,
H64/H96, geomean speedup vs baseline. `--fast` runs a single shape for tight
agent feedback loops (blog ablation: 0.9 min/test beats 1.9 min/test).

    python bench/bench.py --candidate runs/<name> [--fast] [--trace out.json]
Baseline for speedup: runs/baseline-fla.
"""
import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import torch

DEVICE = "cuda"
BASELINE_DIR = Path(__file__).resolve().parent.parent / "runs" / "baseline-fla"


def load_candidate(path):
    spec = importlib.util.spec_from_file_location("candidate", Path(path) / "candidate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_forward()


def workloads():
    """(name, H, cu_seqlens|None) with K=V=128, B=1, ~8192 tokens total (blog set)."""
    w = [
        ("H64_fixed_8192", 64, None),
        ("H96_fixed_8192", 96, None),
        ("H64_varlen_mixed", 64, torch.tensor([0, 100, 917, 2048, 4001, 6144, 8192], dtype=torch.int32)),
        ("H96_varlen_mixed", 96, torch.tensor([0, 100, 917, 2048, 4001, 6144, 8192], dtype=torch.int32)),
        ("H64_varlen_uniform", 64, torch.arange(0, 8193, 1024, dtype=torch.int32)),
        ("H96_varlen_uniform", 96, torch.arange(0, 8193, 1024, dtype=torch.int32)),
    ]
    return w


def gen_inputs(T, H, K, V, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    mk = lambda *s: torch.empty(*s, device=DEVICE).normal_(generator=g)
    q, k, v = mk(1, T, H, K), mk(1, T, H, K), mk(1, T, H, V)
    gg = -mk(1, T, H, K).abs()
    beta = torch.rand(1, T, H, device=DEVICE, generator=g).sigmoid()
    return q.bfloat16(), k.bfloat16(), v.bfloat16(), gg.bfloat16(), beta.bfloat16()


def time_fn(fn, q, k, v, g, beta, cu, reps=10, warmup=3):
    for _ in range(warmup):
        fn(q, k, v, g, beta, cu_seqlens=cu)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(q, k, v, g, beta, cu_seqlens=cu)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]  # median


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--fast", action="store_true", help="single workload (tight feedback loop)")
    ap.add_argument("--trace", default=None, help="export torch profiler chrome trace of baseline-style run")
    args = ap.parse_args()

    cand = load_candidate(args.candidate)
    base = load_candidate(BASELINE_DIR)
    K = V = 128
    wl = workloads()[:1] if args.fast else workloads()

    rows = []
    for name, H, cu in wl:
        cu = cu.to(DEVICE) if cu is not None else None
        T = int(cu[-1]) if cu is not None else 8192
        seed = 777
        q, k, v, g, beta = gen_inputs(T, H, K, V, seed)
        t_base = time_fn(base, q, k, v, g, beta, cu)
        t_cand = time_fn(cand, q, k, v, g, beta, cu)
        sp = t_base / t_cand
        rows.append({"workload": name, "H": H, "tokens": T, "baseline_ms": round(t_base * 1e3, 3),
                     "candidate_ms": round(t_cand * 1e3, 3), "speedup": round(sp, 4)})
        print(f"{name:24s} base {t_base*1e3:8.2f} ms | cand {t_cand*1e3:8.2f} ms | {sp:6.3f}x")

    geo = math.exp(sum(math.log(r["speedup"]) for r in rows) / len(rows))
    print(f"\nGEOMEAN SPEEDUP: {geo:.4f}x  ({len(rows)} workloads)")
    out = {"candidate": args.candidate, "geomean_speedup": round(geo, 4), "rows": rows}
    outp = Path(__file__).resolve().parent.parent / "bench_report.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"report: {outp}")

    if args.trace:
        from torch.profiler import ProfilerActivity, profile
        q, k, v, g, beta = gen_inputs(8192, 64, K, V, 999)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                cand(q, k, v, g, beta, cu_seqlens=None)
            torch.cuda.synchronize()
        prof.export_chrome_trace(args.trace)
        print(f"trace: {args.trace}")


if __name__ == "__main__":
    main()