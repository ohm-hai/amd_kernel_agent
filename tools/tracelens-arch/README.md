<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->
# Architecture Performance Specs

These performance metrics are derived from the following sources:

- [Understanding Peak and Max-Achievable FLOPS](https://rocm.blogs.amd.com/software-tools-optimization/Understanding_Peak_and_Max-Achievable_FLOPS/README.html)
- [AMD Instinct MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [AMD Instinct MI325X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)

They represent peak performance bounds used to estimate the optimization headroom available given measured kernel performance. In practice, real kernels may not reach these bounds.

## MI355X.json (added 2026-10-09)

AMD has not published max-achievable numbers for MI355X, so the `max_achievable_tflops`
entries were measured on one MI355X (gfx950, ROCm 7.0, PyTorch 2.9.0a0) with
`kernel-lab/tools/measure_maf.py`, following the definition in the FLOPs blog above:
random data, warmed-up steady state, operands larger than the 256 MB Infinity Cache, best
rate over large square sizes.

- `matrix_*`: hipBLASLt through `torch.matmul` (fp32 with TF32 disabled),
  `torch._scaled_mm` (fp8 e4m3fn) and `torch._int_mm` (int8), sizes 8192–16384.
- `vector_fp32`, `vector_fp64`: a Triton kernel running four independent FMA chains per lane.
- `vector_fp16`: same kernel. Triton did not emit packed fp16 math (it measured barely
  above fp32), so this understates the hardware rate.
- `vector_bf16`: **not measured.** The same kernel reached only 6.7 TFLOPS because Triton
  emulates bf16 arithmetic through fp32 with software rounding. The value copies
  `vector_fp16`, matching the convention in MI300X.json and MI325X.json.
- `mem_bw_gbps`: AMD's stated "up to 8 TB/s" (ROCm blog on MoE training on MI355X), the same
  datasheet basis as the other files. `torch.copy_` on 4 GiB measured 4.68 TB/s, and FLA's
  KDA decode kernel 5.6 TB/s, from hardware counters.

Re-measure after ROCm or library upgrades.
