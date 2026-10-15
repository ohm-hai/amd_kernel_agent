---
name: mi355x-kernel-skill
description: Profile and optimize GPU kernels on AMD MI355X (gfx950, CDNA4, ROCm). Use when the user asks to profile a kernel, analyze performance, diagnose bottlenecks, write an optimization plan, or compare hardware efficiency on AMD Instinct — including TraceLens/rocprof workflows and KDA evidence-loop integration.
---

# Skill: AMD Kernel Optimization (MI355X / gfx950)

**When to use:** profiling/optimizing any GPU kernel on MI355X; analyzing rocprof or
TraceLens output; planning kernel work in the KDA evidence loop
(`/root/amd-workspace/kernel-lab`, see `AGENTS.md`).

**AMD analog of** `kda/skills/ncu-report-skill` (which targets NVIDIA B200/sm_100 with
Nsight Compute). Everything here is measured on-box unless marked spec.

## Golden rule

**Profile -> Diagnose -> Plan, in that order. Never guess.**

One misdiagnosis costs a full candidate iteration (we burned v3 on a store-pattern
theory that was wrong; the profiler would have said so in 30 seconds).

## Quickstart

0. One run = one directory under the task's `profile/<run_name>/` (harness, reports,
   analysis). Never reuse.
1. Identify the question (which kernel, which shapes, which metric).
2. Get kernel-level truth first (wrapper times LIE — FlyDSL dispatch ~90us, Triton
   ~38us, our Python wrapper 47-133us; see reference/01):
   `python3 -c` torch.profiler with `ProfilerActivity.CUDA` -> `key_averages()` or
   export chrome trace -> `TraceLens_generate_perf_report_pytorch`.
3. TraceLens report gives GPU timeline (idle%!), kernel means, category breakdown:
   `TraceLens_generate_perf_report_pytorch --profile_json_path <t.json>
   --output_xlsx_path r.xlsx --output_csvs_dir csvs`
4. Collect hardware counters with rocprofv3, one derived metric per pass
   (reference/08). Never trust `FETCH_SIZE` on gfx950: it reports half the bytes read.
5. Walk the six dimensions (reference/05). On this box, dispatch/launch, memory
   bandwidth and instruction-issue stalls dominate everything measured so far.
6. Match patterns via the playbook (reference/06); write `REPORT.md` ranked by impact.
7. Feed evidence into the KDA loop with `kernel-lab/tools/record.py`: `results.csv`
   rows and `candidates.jsonl` status. Check the GPU is free first (`tools/env_stamp.py`).

## File index

| File | Purpose |
|---|---|
| `reference/00-mi355x-architecture.md` | gfx950 hardware facts (verified on-box) + what each means for kernels |
| `reference/01-toolchain-and-traps.md` | FlyDSL/Triton/TraceLens idioms, measured dispatch costs, known faults |
| `reference/04-measured-baselines.md` | campaign numbers: KDA decode/prefill, FLA, dispatch, BW ceilings |
| `reference/05-analysis-dimensions.md` | six dimensions mapped to rocprofv3/TraceLens metrics |
| `reference/06-diagnosis-playbook.md` | signal -> cause -> fix: KDA patterns A-N translated to AMD, plus our measured cases |
| `reference/08-mi355x-metric-names.md` | verified gfx950 counter names, Nsight crosswalk, collection recipes, gotchas |

Numbering of 05, 06 and 08 matches `kda/skills/ncu-report-skill/reference/`, so the
same file number covers the same topic on both vendors.

## Hardware quick table (verified: rocminfo; spec values marked)

| Resource | Value | Kernel implication |
|---|---|---|
| CUs | 256 = 8 XCDs x 32 (counter dimension `DIMENSION_XCC [0:8]`) | need >=~512 workgroups to hide latency |
| SIMDs | 4 per CU; up to 8 resident waves each (32/CU, per `OccupancyPercent`) | one wave issues per SIMD per cycle |
| Wavefront | 64 lanes (not 32!) | shuffle_xor trees span 64; cross-wave = LDS |
| Clock | <= 2.4 GHz | FP32 VALU peak ~157 TFLOP/s (spec) |
| HBM3E | 288 GB, ~8 TB/s (spec) | THE wall for state-round-trip kernels |
| L1/L2/L3 | 32 KB vL1D per CU / 4 MB L2 per XCD (32 MB total) / 256 MB Infinity Cache | redundant same-addr loads are cheap; L2 is not shared across XCDs |
| LDS | 160 KB per CU, all allocatable by one workgroup (HIP `sharedMemPerBlock`) | swizzle trees need no explicit LDS alloc |
| MFMA | BF16/FP8/MXFP8/FP4 | matrix ops; GEMV wastes 14/16 M-rows — avoid |
