# Diagnosis Playbook (signal -> cause -> fix) — MI355X edition

AMD analog of `kda/skills/ncu-report-skill/reference/06-diagnosis-playbook.md`.
Read it after collecting metrics (`05-analysis-dimensions.md`); counter names and
collection recipes are in `08-mi355x-metric-names.md`.

- **Part 1** — cases actually hit in the KDA decode campaign (kernel-lab/kda-serving).
- **Part 2** — the KDA playbook's Patterns A–N translated to gfx950 signals and fixes.
- **Part 3** — a full worked diagnosis (case M1: v7 vs FLA decode).
- **Part 4** — ranking template for the final report.

Most kernels match 2–4 patterns at once. Rank them by measured magnitude and fix
the biggest first.

---

## Part 1 — Measured campaign cases

| Signal (measurement) | Likely cause | Fix / lesson |
|---|---|---|
| TraceLens gpu_timeline idle% >50% on short kernels | dispatch/launch bound | CUDA graphs; compare graphed-vs-graphed; never compare raw wall across toolchains |
| Wall ~= kernel time at large B, but TB/s well below FLA's 5.6 | issue stalls, BW efficiency, or dtype | measure first (case M1): `SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES`, `VmemLatency`, write efficiency `TCC_EA0_WRREQ_64B_sum / TCC_EA0_WRREQ_sum`; state dtype (fp32 -> bf16/fp16 halves traffic) |
| Wall >> kernel sum at small B | CPU dispatch | graphs; hoist launchers; pre-allocate outputs; measure kernel-only via profiler |
| Kernel correct at step 0, diverges with carried state | state math bug (sign, missing term) | beta=0 probe isolates update vs decay; identity probe (q=k=e0) isolates normalization; extract effective intermediate analytically from outputs (v3 kv-sign case) |
| Output ~= 0 or garbage only for some lanes | dtype mismatch on load (bf16 read as f32) | vector extf with matching vector type (v5 case) |
| Kernel slow ~2-3x with all-lanes-identical reductions | redundant per-lane tree work | distribute: column-groups per wavefront, rows paired in-lane (v5 -> v6: 900 -> 200 ops/lane, gap 2.6->1.5x) |
| Store-pattern optimization = no perf change | write-combining absorbs it | v3 lesson, now counter-backed: v7 already issues 98.5% full 64 B writes (FLA 100%) |
| Speedup >1.5x on weak tests | verifier hole (blog hacks A-E) | fp64 reference + stress probes (deep decay ~600 bits, beta extremes, boundary T) BEFORE promotion |
| torch.compile/inductor crash (cluster_dims) | toolchain skew | don't use torch.compile on this box (triton 3.4 vs torch 2.9a0) |
| GPU memory access fault on big tensors | FLA int32 indexing >2^31 elems | cap B*T*H*K; 64-bit indexing in candidate kernels |
| Bandwidth from `FETCH_SIZE` looks impossibly low (~half of expected reads) | `FETCH_SIZE` miscounts 128 B reads on gfx950 | use `TCC_EA0_RDREQ_sum * 128`; calibrated on a 1 GiB copy (08, Gotcha 1) |
| rocprofv3 prints error code 38 and never exits | counter set does not fit one pass; tool hangs after abort | one derived metric per pass; always wrap in `timeout` (08, Gotcha 3) |

---

## Part 2 — KDA Patterns A–N on gfx950

Each pattern keeps the KDA letter. "Measured here" marks entries with on-box evidence.

### Pattern A — Small grid / CU idle
**Signals:** workgroups (`Grid_Size / Workgroup_Size`) below `CU_NUM` (256), or below
~512 for latency hiding; `MeanOccupancyPerCU` well under `MeanOccupancyPerActiveCU`
(idle CUs pull the first down); `GPU_UTIL` high while `SQ_BUSY_CU_CYCLES` is low.
**Why:** one workgroup runs on one CU. MI355X has 256 CUs across 8 XCDs.
**Fix:** as on NVIDIA, split along an unparallelized dimension (K/V tiles, heads, split-K).
**Measured here:** KDA decode at B=1 launches 384 workgroups (96 heads × 4 column
tiles), about 1.5 per CU, so it is latency-exposed. The NV=4 column split is already
the split dimension; going further needs a K-split with a cross-workgroup reduction.
**Exception:** batch-1 decode is small by nature; graphs (Part 1) matter more than grid.

### Pattern B — Tail effect (variable-length inputs)
**Signals:** kernel-trace duration much longer than typical per-workgroup work;
`SQ_BUSY_CU_CYCLES` / `GRBM_GUI_ACTIVE` low over the dispatch. rocprofv3 has no per-CU
active-cycle distribution, so use `--att` on a target CU for a timeline.
**Fix:** sort/pack by length, split long sequences, or a chunkwise kernel.
**Measured here:** FLA `chunk_kda` shows no tail effect: varlen mixed (6 seqs) and
uniform (8 × 1024) run within about 1% of each other at 8192 tokens (kda-attn bench:
1.75 vs 1.76 ms at H64, 2.54 vs 2.51 ms at H96).

### Pattern C — Uncoalesced global loads
**Signals:** `TA_BUFFER_COALESCEABLE_WAVEFRONTS_sum / TA_BUFFER_READ_WAVEFRONTS_sum` or
`TD_COALESCABLE_WAVEFRONT_sum / TD_LOAD_WAVEFRONT_sum` low; read bytes
(`TCC_EA0_RDREQ_sum * 128`) far above the useful bytes; `TCP_TOTAL_CACHE_ACCESSES` high
per `SQ_INSTS_VMEM_RD`.
**Why:** a wave64 whose lanes touch many 128 B lines pulls whole lines for a few bytes.
**Fix:** contiguous per-lane addresses and dwordx4 (vec4) loads: one wave64 dwordx4 load
covers 1 KB, eight full 128 B lines. For AoS data, switch to SoA.
**Exception:** gathers by random index.

### Pattern D — Sparse writes (low store efficiency)
**Signals:** `TCC_EA0_WRREQ_64B_sum / TCC_EA0_WRREQ_sum` well below 1; `MemWrites32B`
large relative to bytes written.
**Fix:** let contiguous lanes write contiguous data; pack small results into vec4 stores.
**Measured here:** not a problem for KDA decode: FLA 100% and v7 98.5% full 64 B writes.
The remaining 1.5% in v7 are the bf16 `o` stores.

### Pattern E — Latency-bound
Two sub-cases on AMD, separated by `VmemLatency`:
- **E1, memory latency:** `VmemLatency` high and few memory instructions in flight
  (`SQ_INST_LEVEL_VMEM` low per wave). Fix: issue more independent loads earlier,
  unroll, raise occupancy. There is no cp.async/TMA; the AMD tools are early
  `buffer_load` issue plus direct-to-LDS buffer loads (usage shows up in
  `TA_BUFFER_READ_LDS_WAVEFRONTS_sum`).
- **E2, dependency/issue latency:** `SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES` high while
  `VmemLatency` is normal or low and bandwidth is below peak. Waves wait on chains of
  dependent instructions (e.g. LDS-path swizzles), not on DRAM. Fix: interleave
  independent chains, hoist loads above dependent math, and do the intra-row part of
  sum reductions with DPP (`row_shr`, `row_bcast:15/31` on gfx9-class GPUs) instead of
  LDS-path swizzles. gfx9 DPP has no xor-mask mode, so exact xor trees still need swizzles.
Confirm which instructions stall with stochastic PC sampling (08).
**Measured here:** v7 decode is E2 (case M1).

### Pattern F — Compute-bound but not on matrix cores
**Signals:** `VALUBusy` high, `SQ_INSTS_VALU_FLOPS_FP32` large, `MfmaUtil` ≈ 0 on
GEMM-like work.
**Fix:** MFMA via Triton `tl.dot`, Composable Kernel, or hipBLASLt.
**Exception:** GEMV (M ≤ 2). MFMA pads M to 16 and runs at ~1/8 efficiency, so VALU
wins. KDA decode is GEMV-shaped: its VALU use is ~1% of peak and it is not
compute-bound.

### Pattern G — Atomics contention
**Signals:** `TCC_ATOMIC` and `TA_BUFFER_ATOMIC_WAVEFRONTS_sum` large while compute
counters stay low.
**Fix:** reduce within the wave (DPP / swizzle), then within the workgroup (LDS), then
one atomic per workgroup.

### Pattern H — LDS bank conflicts
**Signals:** `SQ_LDS_BANK_CONFLICT` a large fraction of `SQ_ACTIVE_INST_LDS`;
`LDSBankConflict` (% of GPU time) high; `SQ_LDS_ADDR_CONFLICT` for address conflicts.
**Fix:** padding or XOR swizzle of LDS indices, as on NVIDIA.
**Note:** gfx950 has 160 KB of LDS per CU (08), so deeper LDS staging is possible
than on MI300X's 64 KB.

### Pattern I — Synchronization overhead
**Signals:** no aggregate barrier-stall counter in rocprofv3. Use PC sampling and look
for samples on `s_barrier`; `SPI_RA_BAR_CU_FULL_CSN` non-zero means barrier resources
limited wave launch.
**Fix:** keep reductions inside one wave64 and use wave-scoped exchange instead of
`s_barrier`.
**Measured here:** v4 used 6 barriers per token and ran at 2.6–3.3× FLA's kernel time;
v6 uses none (in-wave trees) and runs at 1.5–1.7×. The gain came from removing the
barriers together with redistributing the tree work (v5 -> v6), so it can't be
attributed to barriers alone.

### Pattern J — Low achieved vs theoretical occupancy
**Signals:** `OccupancyPercent` low. If `SPI_RA_VGPR_SIMD_FULL_CSN`,
`SPI_RA_LDS_CU_FULL_CSN` and `SPI_RA_WAVE_SIMD_FULL_CSN` are zero, resources are not
the limit and the cause is imbalance, a short kernel, or stalls (Patterns B, E, I).
If one is non-zero, that resource limits launch.
**Reference:** theoretical maximum is 8 waves per SIMD, 32 per CU (agent info).

### Pattern K — Register spill
**Signals:** `Scratch_Size` > 0 in the counter CSV; high `VGPR_Count` plus
`Accum_VGPR_Count`.
**Fix:** bound registers (`__launch_bounds__`, `amdgpu_waves_per_eu`, Triton
`num_warps`); reduce live values. On CDNA, MFMA accumulators can live in AGPRs
(`Accum_VGPR_Count`), which frees architectural VGPRs.
**Measured here:** FLA 44 VGPR and v7 32 VGPR, both with `Scratch_Size` 0.

### Pattern L — FP64 used unintentionally
**Signals:** `SQ_INSTS_VALU_FLOPS_FP64` or `SQ_INSTS_VALU_FMA_F64` / `SQ_INSTS_VALU_ADD_F64`
non-zero in a kernel meant to be fp32.
**Fix:** `f` suffix on literals; fp32 transcendentals.

### Pattern M — Pipeline bubbles (no compute/memory overlap)
**Signals:** neither `VALUBusy` nor achieved bandwidth near its ceiling,
`SQ_ACTIVE_INST_VMEM` low and `SQ_WAIT_INST_ANY` high. rocprofv3 has no PM-sampling
time series; use `rocprofv3 --att` (thread trace) on one CU to see the alternation.
**Fix:** double-buffer through LDS (160 KB allows more stages), software pipelining,
direct-to-LDS loads.

### Pattern N — Wave divergence
**Signals:** `VALUUtilization` (active lanes per VALU instruction, out of 64) well below
100%; many `SQ_INSTS_BRANCH`.
**Fix:** as on NVIDIA. Wave64 makes each divergent branch twice as wide as a warp32.
**Exception:** the tail of in-wave reduction trees.

---

## Part 3 — Case M1: v7 vs FLA decode (Kimi K3 KDA, B=128, fp32 state)

**Question:** both kernels move the same data. Why is v7 1.65× slower?

**Step 1, rule out bandwidth layout.** Reads 850 MB vs 850 MB, writes 825 MB vs 809 MB,
full 64 B writes 98.5% vs 100%. Not a layout problem (Patterns C, D excluded).

**Step 2, rule out occupancy.** Same 196,608 waves, `Scratch_Size` 0, v7 uses fewer
VGPRs (32 vs 44). Not a resource limit (Patterns J, K excluded).

**Step 3, locate the time.** Total `SQ_WAVE_CYCLES`: v7 1.93 G vs FLA 0.68 G, so v7
waves live 2.8× longer. `SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES`: v7 74% vs FLA 27%.

**Step 4, separate E1 from E2.** `VmemLatency`: v7 1,848 cycles vs FLA 2,597. v7's
memory instructions finish faster, so it is not waiting on DRAM. It is Pattern E2.

**Diagnosis:** v7 waves sit idle waiting to issue the next dependent instruction.
v7 runs 19 shuffle trees (3 for L2 norms, 8 for kv, 8 for od) of 6 dependent steps
each, and issues its state loads only after the norm trees.

**Next fixes (v8 candidates), in order:**
1. Issue the state loads before the norm trees, since they don't depend on them.
2. Interleave the 16 kv/od trees step by step, so independent swizzles overlap.
3. Do the intra-row reduction steps with DPP (`row_shr`, `row_bcast`) instead of
   LDS-path swizzles (sum reductions don't need the exact xor pattern).

**Verify before building:** run stochastic PC sampling on v7 and confirm the samples
concentrate on swizzle/waitcnt instructions.

---

## Part 4 — Ranking template for the final report

rocprofv3 has no Nsight-style "Est. Speedup". Rank by measured counter gap ×
effort, and state the counter values that justify each item:

```
Priority 1: <pattern> — <concrete fix>
  Evidence: <counter values, ours vs reference kernel>
  Expected gain: <reasoning from the gap, e.g. wait-for-issue 74% -> 27%>
  Effort: <low / medium / high>
  Why now: <reason this is the highest-leverage fix>

Priority 2: ...
```

Keep it to 3–5 priorities. Feed accepted plans into the KDA loop (`docs/draft.md`) and
results into `results.csv` / `candidates.jsonl` through `kernel-lab/tools/record.py`.
