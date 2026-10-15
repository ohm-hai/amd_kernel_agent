#!/usr/bin/env python3
"""Measure the per-launch floor of graph-replayed kernels on the current GPU.

A decode step can't be faster than an empty kernel with the same launch shape, so this
bounds the small-batch speedup any candidate can reach.

    HIP_VISIBLE_DEVICES=<free gpu> python3 tools/launch_floor.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _empty(X):
    if tl.program_id(0) < 0:
        tl.store(X, 0.0)


def graphed_us(fn, steps=8, reps=200):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(steps):
            fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / (reps * steps)


if __name__ == "__main__":
    x = torch.zeros(1, device="cuda")
    for grid in (1, 384, 3072, 49152):
        us = graphed_us(lambda: _empty[(grid,)](x, num_warps=4))
        print(f"empty kernel, {grid:6d} workgroups x 256 threads: {us:5.2f} us per launch (graph replay)")
