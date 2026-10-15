---
id: migration-sm100-to-gfx950
title: "Porting Blackwell (SM100) Kernel Practice to AMD gfx950"
type: migration
from_arch: sm100
to_arch: gfx950
tags: [wave64, lds, mfma, dpp, tcgen05, tma, cuda-graphs, triton]
confidence: inferred
related: [hw-gfx950-cdna4, hw-gfx950-perf-counters, hw-tcgen05-mma, hw-tma, hw-tmem, lang-triton, kernel-kda-decode-mi355x]
sources: [blog-kernel-lab-mi355x, blog-amd-primus-moe, blog-amd-tracelens]
blackwell_relevance: "Starts from Blackwell concepts in this wiki and gives the gfx950 counterpart for each, so Blackwell-tuned kernels can be ported and re-profiled."
---

## Concept map

| Blackwell (SM100) | gfx950 (MI355X) | Porting note |
|---|---|---|
| SM, 148 per GPU | CU, 256 per GPU (8 XCDs × 32) | grids need ≥256 workgroups to fill the GPU |
| warp = 32 threads | wavefront = 64 lanes | reductions take 6 in-wave steps; divergence is twice as wide |
| `__shfl_xor_sync` | `ds_swizzle` / `ds_bpermute`, or DPP for sum steps | swizzles use the LDS pipe; dependent chains stall issue (kernel-kda-decode-mi355x) |
| shared memory (up to 228 KB per SM) | LDS, 160 KB per CU, all usable by one workgroup | larger than CDNA3's 64 KB |
| tcgen05 + TMEM | MFMA with AGPR accumulators | no separate tensor memory |
| TMA bulk async copy | early `buffer_load`, direct-to-LDS buffer loads | no hardware descriptor-driven bulk copy |
| clusters, CLC, 2-SM MMA | no equivalent | persistent kernels use atomic work queues |
| Nsight Compute | rocprofv3 counters + PC sampling + `--att` | names and traps in hw-gfx950-perf-counters |
| CUDA graphs | HIP graphs (`torch.cuda.CUDAGraph` works on ROCm) | as important as on NVIDIA: small-B decode is dispatch-bound |

## Language support on this stack

- **Triton** runs on gfx950 (pytorch-triton-rocm 3.4.0). FLA's KDA kernels work unchanged.
- **FlyDSL** (MLIR DSL used by Primus-Turbo) compiles kernels for gfx950; per-launch
  dispatch measured about 90 µs, so capture launches in graphs.
- **CuTe DSL / tcgen05 PTX** have no AMD path; port the algorithm, not the code.

## Porting checklist

1. Re-derive the thread-to-data mapping for wave64 before tuning anything else.
2. Re-check occupancy limits: 8 waves per SIMD, 160 KB LDS, `VGPR_Count` from rocprofv3.
3. Replace TMA/TMEM pipelines with buffer-load staging through LDS.
4. Re-baseline with graphs on both sides; raw wall-clock comparisons across toolchains
   mostly measure dispatch.
5. Validate bandwidth claims with `TCC_EA0_RDREQ_sum * 128`, not `FETCH_SIZE`.

```python
import torch
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for _ in range(8):
        out = step(inputs)   # capture several decode steps per graph
graph.replay()
```
