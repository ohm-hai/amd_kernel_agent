---
id: blog-amd-primus-moe
title: "MoE Training Best Practices on AMD GPUs (Primus)"
author: "Wen Xie, Xiaoming Peng, Xiaobo Chen, et al. (AMD)"
url: https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html
source_category: benchmark-blog
architectures: [gfx950]
tags: [moe, grouped-gemm, mfma]
hardware_features: [mfma]
kernel_types: [moe, grouped-gemm]
languages: [python]
retrieved_at: 2026-10-08
description: "AMD ROCm blog (2025-12-16) on MoE training with Primus / Primus-Turbo on MI300X and MI355X; source for MI355X memory specs and kernel-launch tuning."
---

# MoE Training Best Practices on AMD GPUs (AMD ROCm blog)

Kernel-relevant content only. The blog also covers distributed-training topics
(expert-parallel dispatch, pipeline schedules), which are outside this wiki's scope.

## Hardware facts stated by the blog

- MI300X: 192 GB HBM3, peak memory bandwidth about 5.325 TB/s.
- MI355X: 288 GB HBM3E, bandwidth up to 8 TB/s.
- Benchmarks ran on 8× MI355X with dual AMD EPYC 9575F, ROCm 7.1, PyTorch 2.9.0.

## Kernel-level techniques described

- **Grouped GEMM**: Primus-Turbo uses a fused Composable Kernel (CK) grouped GEMM that
  processes all experts in one kernel launch, replacing a multi-stream approach that
  still suffered scheduling overhead. It can also pick the fastest grouped-GEMM backend
  separately for forward and backward kernels.
- **Kernel-argument pool size**: MoE layers launch many small kernels. An undersized HIP
  kernel-argument pool stalls launches; the blog raises it to 12 MB with
  `HSA_KERNARG_POOL_SIZE=12582912`.
- **Sync-free MoE**: removes device-to-host synchronizations (router, permutation,
  grouped MLP) so kernel launches are not blocked by CPU waits.
- **Loss fusion**: fuses the large-vocabulary loss into one kernel to cut memory and
  launch overhead.

## Profiling workflow recommended

Torch Profiler traces viewed in Perfetto, then TraceLens (`blog-amd-tracelens`) for
hierarchical breakdowns, roofline metrics and trace comparison.
