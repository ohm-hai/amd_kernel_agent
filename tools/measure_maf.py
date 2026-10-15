#!/usr/bin/env python3
"""Measure max-achievable FLOPs (MAF) and copy bandwidth on the current GPU.

Method follows AMD's definition (rocm.blogs.amd.com "Understanding Peak, Max-Achievable
& Delivered FLOPs"): realistic random data, warmed-up steady state, operands larger than
the 256 MB Infinity Cache, best rate over a sweep of large sizes.

    HIP_VISIBLE_DEVICES=<free gpu> python3 tools/measure_maf.py > maf.json
"""
import json
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from env_stamp import env_stamp, require_free_gpu  # noqa: E402

DEV = "cuda"


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(3):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters / 1e3)
    return best


def gemm_tflops(dtype, sizes):
    best = 0.0
    for n in sizes:
        a = torch.randn(n, n, device=DEV).to(dtype)
        b = torch.randn(n, n, device=DEV).to(dtype)
        t = timed(lambda: a @ b, iters=10)
        best = max(best, 2 * n ** 3 / t / 1e12)
        del a, b
    return best


def fp8_tflops(sizes):
    best = 0.0
    for n in sizes:
        a = torch.randn(n, n, device=DEV).to(torch.float8_e4m3fn)
        b = torch.randn(n, n, device=DEV).to(torch.float8_e4m3fn).t()  # column-major B
        one = torch.tensor(1.0, device=DEV)
        f = lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        best = max(best, 2 * n ** 3 / timed(f, iters=10) / 1e12)
    return best


def int8_tops(sizes):
    best = 0.0
    for n in sizes:
        a = torch.randint(-64, 64, (n, n), device=DEV, dtype=torch.int8)
        b = torch.randint(-64, 64, (n, n), device=DEV, dtype=torch.int8)
        best = max(best, 2 * n ** 3 / timed(lambda: torch._int_mm(a, b), iters=10) / 1e12)
    return best


@triton.jit
def _fma_chain(X, Y, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    a = x * 0.0 + 0.999
    c = x * 0.0 + 0.001
    y0, y1, y2, y3 = x, x + 1, x + 2, x + 3
    for _ in range(ITERS):
        y0 = y0 * a + c
        y1 = y1 * a + c
        y2 = y2 * a + c
        y3 = y3 * a + c
    tl.store(Y + offs, y0 + y1 + y2 + y3)


def vector_tflops(dtype, n=16 * 1024 * 1024, iters=1024, block=1024):
    x = torch.rand(n, device=DEV).to(dtype)
    y = torch.empty_like(x)
    f = lambda: _fma_chain[(n // block,)](x, y, ITERS=iters, BLOCK=block)
    return 2 * 4 * n * iters / timed(f, iters=5) / 1e12


def copy_tbps(n_bytes=4 * 1024 ** 3):
    src = torch.randn(n_bytes // 4, device=DEV)
    dst = torch.empty_like(src)
    return 2 * n_bytes / timed(lambda: dst.copy_(src), iters=10) / 1e12


def main():
    require_free_gpu()
    big = [8192, 12288, 16384]
    out = {"env": env_stamp(), "max_achievable_tflops": {}, "measured_copy_tb_per_s": None, "errors": {}}
    m = out["max_achievable_tflops"]
    jobs = {
        "matrix_bf16": lambda: gemm_tflops(torch.bfloat16, big),
        "matrix_fp16": lambda: gemm_tflops(torch.float16, big),
        "matrix_fp32": lambda: gemm_tflops(torch.float32, [8192, 12288]),
        "matrix_fp64": lambda: gemm_tflops(torch.float64, [8192]),
        "matrix_fp8": lambda: fp8_tflops(big),
        "matrix_int8": lambda: int8_tops(big),
        "vector_fp32": lambda: vector_tflops(torch.float32),
        "vector_fp16": lambda: vector_tflops(torch.float16),
        "vector_bf16": lambda: vector_tflops(torch.bfloat16),
        "vector_fp64": lambda: vector_tflops(torch.float64, iters=256),
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    for key, job in jobs.items():
        t0 = time.perf_counter()
        try:
            m[key] = round(job(), 1)
        except Exception as e:  # report and continue; missing entries are flagged in the JSON
            out["errors"][key] = f"{type(e).__name__}: {e}"[:300]
        print(f"{key:12s} {m.get(key, 'ERROR')!s:>10}  ({time.perf_counter() - t0:.0f}s)", file=sys.stderr, flush=True)
    out["measured_copy_tb_per_s"] = round(copy_tbps(), 2)
    print(f"copy bandwidth {out['measured_copy_tb_per_s']} TB/s", file=sys.stderr)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
