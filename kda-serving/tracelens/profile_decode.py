#!/usr/bin/env python3
"""Profile decode candidates through named ops and emit a trace for TraceLens.

    HIP_VISIBLE_DEVICES=<free gpu> python3 tracelens/profile_decode.py \
        --candidates baseline-fla-decode flydsl-decode-v7 --B 128 --out profile/<run>/trace.json
    TraceLens_generate_perf_report_pytorch --profile_json_path profile/<run>/trace.json \
        --extension_file tracelens/kda_extension.py --gpu_arch_platform MI355X \
        --output_xlsx_path profile/<run>/report.xlsx --output_csvs_dir profile/<run>/csvs
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "tools"))
import inputs as inp  # noqa: E402
from env_stamp import require_free_gpu  # noqa: E402
from kda_ops import decode_op  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    require_free_gpu()
    q, k, v, g, beta = inp.make_inputs(args.B, 1, H=args.H, seed=3, gates="kimi")
    ops = {c: decode_op(c) for c in args.candidates}
    states = {}
    for c in args.candidates:
        mod_dtype = torch.float32 if "bf16" not in c and "fp16" not in c else (
            torch.bfloat16 if "bf16" in c else torch.float16)
        states[c] = torch.zeros(args.B, args.H, 128, 128, dtype=mod_dtype, device="cuda")
        for _ in range(3):
            ops[c](q, k, v, g, beta, states[c])
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for c in args.candidates:
            for _ in range(args.steps):
                ops[c](q, k, v, g, beta, states[c])
        torch.cuda.synchronize()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(args.out)
    print(f"trace: {args.out}")


if __name__ == "__main__":
    main()
