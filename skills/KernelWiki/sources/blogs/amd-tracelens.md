---
id: blog-amd-tracelens
title: "TraceLens: Democratizing AI Performance Analysis"
author: "Adeem Jassani, Gabriel Weisz, Spandan More, Deval Shah, Anshu Raina, Steven K. Reinhardt (AMD)"
url: https://rocm.blogs.amd.com/software-tools-optimization/tracelens/README.html
source_category: benchmark-blog
architectures: [gfx950, sm100]
tags: [python]
languages: [python]
retrieved_at: 2026-10-08
description: "AMD ROCm blog (2026-04-27) on TraceLens, an open-source analyzer for PyTorch/JAX profiler traces that works on any backend the PyTorch profiler supports (ROCm, CUDA)."
---

# TraceLens (AMD ROCm blog)

TraceLens consumes framework profiler traces (PyTorch, JAX) and produces structured
reports. Because it works on profiler traces rather than hardware counters, it runs
on any backend the PyTorch profiler supports, ROCm and CUDA included.

## Capabilities described

- **Trace2Tree**: links Python ops, CPU dispatches and GPU kernels into a tree, exposing
  hidden kernels such as AMP casts and unfused bias adds.
- **GPU timeline breakdown**: computation, exposed communication, exposed memcpy and
  idle time as shares of total time. High idle time means the GPU is waiting (often on
  the CPU).
- **Ops summary by category and by unique input shape**, with kernel time attributed to
  the leaf CPU dispatch so comparisons survive kernel renames.
- **Compute model**: FLOPs and bytes derived from op arguments (e.g. GEMM FLOPs = 2·M·N·K)
  turned into TFLOPS/s and TB/s for roofline analysis, with no reruns or instrumentation.
- **Multi-GPU collective analysis** separating synchronization skew from communication.
- **Trace comparison**, including morphological alignment when call stacks differ.
- **Event replay**: minimal reproducers generated from trace metadata.

## Usage

```bash
pip install git+https://github.com/AMD-AGI/TraceLens.git
TraceLens_generate_perf_report_pytorch --profile_json_path path/to/trace.json
```
