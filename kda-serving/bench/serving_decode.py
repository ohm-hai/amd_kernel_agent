#!/usr/bin/env python3
"""Serving-interface decode benchmark: in-place state pool indexed by slot.

Real serving (vLLM/SGLang) keeps one state pool [P, H, K, V] and updates the B active
sequences' slots in place, via FLA's internal fused_recurrent_kda_fwd with
ssm_state_indices and inplace_final_state=True. The public fused_recurrent_kda wrapper
allocates a fresh state instead, which is what bench/matrix.py measures.

Candidates with native pool support define
    decode_pool(q, k, v, g, beta, pool, idx)   # updates pool[idx] in place
Otherwise they run through gather -> kernel -> scatter, and the extra traffic counts
against them. That wrapper is a stand-in, not a kernel property.

    HIP_VISIBLE_DEVICES=<free gpu> python3 bench/serving_decode.py --candidate runs/flydsl-decode-v7
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd

TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(TASK.parent / "tools"))
import inputs as inp  # noqa: E402
from env_stamp import env_stamp, require_free_gpu  # noqa: E402

K = V = 128
GRAPH_STEPS = 8


def graphed_us_per_step(step, reps=25):
    """Capture GRAPH_STEPS calls of step() in a CUDA graph; return mean us per call."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(GRAPH_STEPS):
            step()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (reps * GRAPH_STEPS)


def load(path):
    spec = importlib.util.spec_from_file_location("cand", Path(path) / "candidate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def candidate_step(mod, q, k, v, g, beta, pool, idx):
    """Build a graph-capturable step for the candidate. Returns (step_fn, mode)."""
    if hasattr(mod, "decode_pool"):
        return (lambda: mod.decode_pool(q, k, v, g, beta, pool, idx)), "native-pool"
    B, _, H, _ = q.shape
    idx_l = idx.long()
    S_in = pool.index_select(0, idx_l)
    if hasattr(mod, "_compile_decode_kernel"):  # FlyDSL convention: raw launcher, static buffers
        launch = mod._compile_decode_kernel()
        S_out = torch.empty_like(S_in)
        o = torch.empty(B, 1, H, V, dtype=torch.bfloat16, device=q.device)
        scale_t = torch.tensor([K ** -0.5], dtype=torch.float32, device=q.device)
        flat = [t.reshape(B * H, -1) for t in (q, k, g, v)] + [beta.reshape(B * H)]

        def step():
            torch.index_select(pool, 0, idx_l, out=S_in)
            launch(flat[0], flat[1], flat[2], flat[3], flat[4], S_in, S_out, o, scale_t,
                   B * H * mod.NV, torch.cuda.current_stream())
            pool.index_copy_(0, idx_l, S_out)
        return step, "gather-scatter-wrapper"
    fwd = mod.get_forward()

    def step():
        torch.index_select(pool, 0, idx_l, out=S_in)
        fwd(q, k, v, g, beta, initial_state=S_in, output_final_state=True)
        pool.index_copy_(0, idx_l, fwd.last_final_state)
    return step, "gather-scatter-wrapper"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--B", type=int, nargs="+", default=[1, 8, 32, 128])
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--gates", default="kimi", choices=["kimi", "synthetic"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    require_free_gpu()
    mod = load(args.candidate) if args.candidate else None
    rows = []
    for B in args.B:
        P = 2 * B
        q, k, v, g, beta = inp.make_inputs(B, 1, H=args.H, seed=17, gates=args.gates)
        idx = torch.randperm(P, device="cuda")[:B].to(torch.int32)
        pool = torch.randn(P, args.H, K, V, device="cuda") * 0.1

        def fla_step():
            fused_recurrent_kda_fwd(q=q, k=k, v=v, g=g, beta=beta, scale=K ** -0.5, initial_state=pool,
                                    inplace_final_state=True, output_final_state=True,
                                    use_qk_l2norm_in_kernel=True, ssm_state_indices=idx)
        row = {"B": B, "H": args.H, "pool_slots": P, "gates": args.gates,
               "fla_native_pool_us": round(graphed_us_per_step(fla_step), 2)}
        if mod is not None:
            pool_c = torch.randn(P, args.H, K, V, device="cuda") * 0.1
            step, mode = candidate_step(mod, q, k, v, g, beta, pool_c, idx)
            row.update(candidate=args.candidate, candidate_mode=mode,
                       candidate_us=round(graphed_us_per_step(step), 2))
            row["speedup_vs_fla"] = round(row["fla_native_pool_us"] / row["candidate_us"], 3)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del pool
        torch.cuda.empty_cache()
    out = {"env": env_stamp(), "graph_steps": GRAPH_STEPS, "rows": rows}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
