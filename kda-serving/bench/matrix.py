#!/usr/bin/env python3
"""Kimi K3 KDA layer serving baseline matrix on MI355X.

H=96, K=V=128 (kimi_k3.yaml). Two modes:
  prefill: FLA chunk_kda over T x B grid (cap 512K tokens/cell)
  decode:  FLA fused_recurrent_kda stepping with state carry, B x warmup-context grid

Outputs baseline_matrix.csv + baseline_matrix.json. Usage:
  CUDA_VISIBLE_DEVICES=6 python3 bench/matrix.py [--quick] [--decode-only|--prefill-only]
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "kda-attn"))
from ref_fp64 import kda_fp64_reference, rel_rmse  # noqa: E402

from fla.ops.kda import chunk_kda, fused_recurrent_kda  # noqa: E402

DEVICE = "cuda"
H, K, V = 96, 128, 128          # Kimi K3 KDA layer
TOK_CAP = 163_840                # per-cell element cap: FLA Triton kernels fault with a
                                 # GPU memory access fault beyond 2^31 elements/tensor
                                 # (int32 indexing overflow; reproduced at B=8 T=32768
                                 # H=96 K=128 = 3.2e9 elems; B=4 T=32768 = 1.6e9 passes)
PREFILL_TS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
PREFILL_BS = [1, 2, 4, 8]
DECODE_BS = [1, 2, 4, 8, 16, 32, 64, 128]
DECODE_CTXS = [1024, 8192, 32768]


GATES = "kimi"  # set from --gates; see inputs.py (timing is gate-independent within 1%)


def gen_inputs(B, T, seed=0, legal=True):
    import inputs as inp
    return inp.make_inputs(B, T, H=H, K=K, V=V, seed=seed, gates=GATES, device=DEVICE)


def prefill_fn():
    def fwd(q, k, v, g, beta, scale=None, initial_state=None):
        kw = dict(use_qk_l2norm_in_kernel=True, output_final_state=True)
        if scale is not None:
            kw["scale"] = scale
        if initial_state is not None:
            kw["initial_state"] = initial_state
        o, S = chunk_kda(q, k, v, g, beta, **kw)
        return o, S
    return fwd


def decode_fn():
    def fwd(q, k, v, g, beta, scale=None, initial_state=None):
        kw = dict(use_qk_l2norm_in_kernel=True, output_final_state=True)
        if scale is not None:
            kw["scale"] = scale
        if initial_state is not None:
            kw["initial_state"] = initial_state
        o, S = fused_recurrent_kda(q, k, v, g, beta, **kw)
        return o, S
    return fwd


def median_time(fn, *args, reps=3, warmup=2):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def correctness_smoke(prefill, decode):
    """Small-T fp64 cross-check for both kernels + 3-step decode stateful carry."""
    B, T = 1, 256
    q, k, v, g, beta = gen_inputs(B, T, seed=5)
    o_ref, S_ref = kda_fp64_reference(q, k, v, g, beta)
    o, S = prefill(q, k, v, g, beta)
    e1, s1 = rel_rmse(o, o_ref), rel_rmse(S, S_ref)
    # decode: 3 steps from zero state, compare against fp64 per-step
    Sd = None
    ok = True
    for i in range(3):
        q1, k1, v1, g1, b1 = (x[:, i:i+1] for x in (q, k, v, g, beta))
        _, S_ref_i = kda_fp64_reference(q[:, :i+1], k[:, :i+1], v[:, :i+1], g[:, :i+1],
                                        beta[:, :i+1])
        o_d, Sd = decode(q1, k1, v1, g1, b1, initial_state=Sd)
        r = rel_rmse(Sd, S_ref_i)
        ok = ok and r < 0.05
        if i == 2:
            s_steps = r
    print(f"smoke: prefill o rRMSE={e1:.4g} state rRMSE={s1:.4g} | decode 3-step state rRMSE={s_steps:.4g}")
    assert e1 < 0.05 and s1 < 0.05 and ok, "correctness smoke FAILED"
    print("smoke: PASS")


def run_prefill(rows, quick=False):
    prefill = prefill_fn()
    for T in PREFILL_TS:
        for B in PREFILL_BS:
            if B * T > TOK_CAP:
                continue
            q, k, v, g, beta = gen_inputs(B, T)
            t = median_time(prefill, q, k, v, g, beta, reps=2 if B * T > 100_000 else 3)
            tps = B * T / t
            rows.append({"mode": "prefill", "impl": "fla-chunk", "B": B, "T": T,
                         "tokens": B * T, "latency_ms": round(t * 1e3, 3),
                         "tokens_per_s": round(tps, 1)})
            print(f"prefill B={B:2d} T={T:6d}: {t*1e3:9.2f} ms | {tps:12.0f} tok/s", flush=True)
            del q, k, v, g, beta
            torch.cuda.empty_cache()
            if quick and T >= 16384:
                break


def run_chunked_prefill(rows):
    """64K context in 8K chunks with state carry vs monolithic."""
    prefill = prefill_fn()
    B, T, CH = 1, 65536, 8192
    q, k, v, g, beta = gen_inputs(B, T, seed=9)
    S = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(0, T, CH):
        sl = slice(s, s + CH)
        _, S = prefill(q[:, sl], k[:, sl], v[:, sl], g[:, sl], beta[:, sl], initial_state=S)
    torch.cuda.synchronize()
    t_chunked = time.perf_counter() - t0
    t_mono = median_time(prefill, q, k, v, g, beta, reps=2)
    rows.append({"mode": "chunked_prefill", "impl": "fla-chunk", "B": B, "T": T,
                 "tokens": B * T, "latency_ms": round(t_chunked * 1e3, 3),
                 "tokens_per_s": round(B * T / t_chunked, 1)})
    print(f"chunked prefill (8x8K, state carry): {t_chunked*1e3:.1f} ms vs monolithic {t_mono*1e3:.1f} ms "
          f"({t_chunked/t_mono:.3f}x)")


def run_decode(rows, quick=False, cand_fwd=None, cand_mod=None, graph_mode=False):
    decode = decode_fn()
    prefill = prefill_fn()

    def warm_state(B, L, seed=13):
        """Recurrent state after prefilling L tokens — chunked with the same
        2^31-element cap as the prefill grid (bounded memory + kernel safety)."""
        CH = min(8192, max(256, TOK_CAP // B))
        S = None
        done = 0
        gq = None
        while done < L:
            q, k, v, g, beta = gen_inputs(B, CH, seed=seed)
            n = CH if L - done >= CH else L - done
            _, S = prefill(q[:, :n], k[:, :n], v[:, :n], g[:, :n], beta[:, :n],
                           initial_state=S)
            done += n
            del q, k, v, g, beta
            torch.cuda.empty_cache()
        return S

    GRAPH_STEPS = 8
    def graph_time_fla(q1, k1, v1, g1, b1, S0, N):
        """Capture GRAPH_STEPS alternating-state FLA calls; replay N/GRAPH_STEPS times."""
        S_a = S0.clone(); S_b = torch.empty_like(S0)
        g = torch.cuda.CUDAGraph()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):  # warmup on side stream
                o_w, S_b = fused_recurrent_kda(q1, k1, v1, g1, b1, initial_state=S_a, **{k: v for k, v in dict(use_qk_l2norm_in_kernel=True, output_final_state=True).items()})
                S_a, S_b = S_b, S_a
        torch.cuda.current_stream().wait_stream(st)
        with torch.cuda.graph(g):
            for _ in range(GRAPH_STEPS):
                o_w, S_new = fused_recurrent_kda(q1, k1, v1, g1, b1, initial_state=S_a, output_final_state=True)
                S_a, S_b = S_new, S_a
        torch.cuda.synchronize()
        reps = max(1, N // GRAPH_STEPS)
        t0 = time.perf_counter()
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / (reps * GRAPH_STEPS)

    def graph_time_cand(launch, qf, kf, gf, vf, bf, S0, s_out, o, scale_t, BHV, N):
        S_a = S0.clone(); S_b = torch.empty_like(S0)
        g = torch.cuda.CUDAGraph()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                launch(qf, kf, gf, vf, bf, S_a, s_out, o, scale_t, BHV, torch.cuda.current_stream())
                S_a, S_b = S_b, S_a
        torch.cuda.current_stream().wait_stream(st)
        with torch.cuda.graph(g):
            for _ in range(GRAPH_STEPS):
                launch(qf, kf, gf, vf, bf, S_a, s_out, o, scale_t, BHV, torch.cuda.current_stream())
                S_a, S_b = S_b, S_a
        torch.cuda.synchronize()
        reps = max(1, N // GRAPH_STEPS)
        t0 = time.perf_counter()
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / (reps * GRAPH_STEPS)

    for L in DECODE_CTXS:
        for B in DECODE_BS:
            S0 = warm_state(B, L)
            q1, k1, v1, g1, b1 = gen_inputs(B, 1, seed=13)
            N = 20 if quick else 100
            S = S0
            if graph_mode:
                per_step = graph_time_fla(q1, k1, v1, g1, b1, S0, N)
            else:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(N):
                    _, S = decode(q1, k1, v1, g1, b1, initial_state=S)
                    S = S.clone()
                torch.cuda.synchronize()
                per_step = (time.perf_counter() - t0) / N
            rows.append({"mode": "decode", "impl": "fla-recurrent", "B": B, "T": 1,
                         "context_L": L, "tokens": B, "latency_ms": round(per_step * 1e3, 4),
                         "tokens_per_s": round(B / per_step, 1)})
            print(f"decode ctx={L:6d} B={B:3d}: fla {per_step*1e3:7.3f} ms/step | {B/per_step:9.0f} tok/s", flush=True)
            if cand_fwd is not None:
                if graph_mode:
                    launch_c = cand_mod._compile_decode_kernel()
                    s_out_c = torch.empty(B, H, K, V, dtype=torch.float32, device=DEVICE)
                    o_c = torch.empty(B, 1, H, V, dtype=torch.bfloat16, device=DEVICE)
                    st_c = torch.tensor([K ** -0.5], dtype=torch.float32, device=DEVICE)
                    qf = q1.reshape(B * H, K).contiguous()
                    kf = k1.reshape(B * H, K).contiguous()
                    gf = g1.reshape(B * H, K).contiguous()
                    vf = v1.reshape(B * H, V).contiguous()
                    bf = b1.reshape(B * H).contiguous()
                    pc = graph_time_cand(launch_c, qf, kf, gf, vf, bf, S0,
                                         s_out_c, o_c, st_c, B * H * cand_mod.NV, N)
                    rows.append({"mode": "decode", "impl": "cand-graph", "B": B, "T": 1,
                                 "context_L": L, "tokens": B,
                                 "latency_ms": round(pc * 1e3, 4),
                                 "tokens_per_s": round(B / pc, 1),
                                 "speedup_vs_fla": round(per_step / pc, 3),
                                 "speedup_vs_fla_graphed": round(per_step / pc, 3)})
                    print(f"decode ctx={L:6d} B={B:3d}: GRAPH cand {pc*1e3:7.3f} ms/step | "
                          f"{B/pc:9.0f} tok/s | {per_step/pc:.2f}x vs graphed-fla", flush=True)
                    del s_out_c, o_c
                    torch.cuda.empty_cache()
                    if quick and B >= 16:
                        break
                    continue
                S = S0
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(N):
                    res = cand_fwd(q1, k1, v1, g1, b1, initial_state=S,
                                   output_final_state=True)
                    S = (res[1] if isinstance(res, tuple)
                         else cand_fwd.last_final_state).clone()
                torch.cuda.synchronize()
                pc = (time.perf_counter() - t0) / N
                rows.append({"mode": "decode", "impl": "flydsl-decode", "B": B, "T": 1,
                             "context_L": L, "tokens": B, "latency_ms": round(pc * 1e3, 4),
                             "tokens_per_s": round(B / pc, 1),
                             "speedup_vs_fla": round(per_step / pc, 3)})
                print(f"decode ctx={L:6d} B={B:3d}: flydsl {pc*1e3:7.3f} ms/step | {B/pc:9.0f} tok/s | {per_step/pc:.2f}x vs fla", flush=True)
            del S0, S
            torch.cuda.empty_cache()
            if quick and B >= 16:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--prefill-only", action="store_true")
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--decode-candidate", default=None,
                    help="runs/<name> with get_forward() decode impl to race vs fla")
    ap.add_argument("--gates", default="kimi", choices=["kimi", "synthetic"],
                    help="input gate distribution (inputs.py); 'synthetic' reproduces pre-2026-10-09 runs")
    ap.add_argument("--graph", action="store_true",
                    help="CUDA-graph the decode loop (kills dispatch overhead; both sides graphed)")
    args = ap.parse_args()
    global GATES
    GATES = args.gates
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tools"))
    from env_stamp import require_free_gpu
    require_free_gpu()
    torch.manual_seed(0)
    prefill, decode = prefill_fn(), decode_fn()
    correctness_smoke(prefill, decode)
    rows = []
    if not args.decode_only:
        run_prefill(rows, quick=args.quick)
        if not args.quick:
            run_chunked_prefill(rows)
    if not args.prefill_only:
        cand_fwd = None
        graph_mode = args.graph
        if args.decode_candidate:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "cand", Path(args.decode_candidate) / "candidate.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cand_fwd = mod.get_forward()
            cand_mod = mod
        run_decode(rows, quick=args.quick, cand_fwd=cand_fwd, cand_mod=cand_mod,
                   graph_mode=args.graph)
    out = Path(__file__).resolve().parent.parent
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out / "baseline_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (out / "baseline_matrix.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out/'baseline_matrix.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()