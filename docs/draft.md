# Draft — bf16-gemm-mi355x (KDA × TraceLens)

## Current baseline and validation

- Baseline `baseline-eager`: 8-block toy transformer (D=1024, V=8192, bf16 GEMMs),
  eager PyTorch, AdamW, single MI355X. Trace + TraceLens report exist:
  `/root/amd-workspace/traces/demo_train_trace.json`, `/root/amd-workspace/reports/train_csvs/`.
- Baseline numbers: computation 61.22 ms/step, idle 1.39%, GEMM = 63.6% of kernel
  time (147 kernels), SDPA fwd+bwd ≈ 17.5%.
- Validation: bf16 allclose (atol 1e-1 rtol 1e-2) vs eager with shared weights;
  per-candidate `runs/<name>/candidate.py validate`.

## Risks and unknowns

- **Environment (confirmed):** primus_turbo `flash_attn_func` → aiter JIT build broken
  (aiter 0.1.9 vs installed composable_kernel mismatch). torch.compile → inductor/triton
  skew (`KernelMetadata.cluster_dims`). Both paths currently unusable; candidates must
  be pure eager or Triton (Triton untested inductor-free).
- Profiler capture is single-step; run-to-run variance is a few percent — differences
  <5% are treated as noise.
- Roofline tables: use `--gpu_arch_platform MI355X` (added 2026-10-09; earlier reports
  used MI300X specs and overstate roofline percentages).

## Candidate directions (ranked)

1. **norm-once** (done, revised): compute `ln1(x)` once instead of 3×/block. Low risk,
   small gain (-2.1% compute, idle 1.39%→0.57%). Kept as new base.
2. **SDPA packed-QKV**: replace MultiheadAttention's module machinery with direct
   `F.scaled_dot_product_attention` on prepacked qkv (removes python/indirection
   overhead, possibly fuses). Medium value.
3. **Triton fused LN+GELU**: hand Triton kernels for the elementwise chains —
   targets the elementwise tail (227 kernels, ~8%). Medium risk (no inductor fallback).
4. **primus_turbo grouped GEMM / flash attention**: highest ceiling (verified working
   for grouped GEMM) but blocked on aiter JIT; revisit after aiter/CK fix.

## Next concrete steps

- Candidate 2 (SDPA packed-QKV): implement in `runs/sdpa-packed/`, validate, profile,
  TraceLens-compare vs `norm-once`.
- If <5% delta, record as noise and move to candidate 3.

## Commands

- Validate: `CUDA_VISIBLE_DEVICES=6 python3 runs/<c>/candidate.py validate`
- Profile:  `CUDA_VISIBLE_DEVICES=6 python3 runs/<c>/candidate.py profile runs/<c>/trace.json`
- Evaluate: `TraceLens_generate_perf_report_pytorch --profile_json_path runs/<c>/trace.json --output_xlsx_path profile/<c>/perf_report.xlsx --output_csvs_dir profile/<c>/csvs`
- Promotion evidence: GEMM TFLOPS/s (compute-metrics), gpu_timeline rows, ops_summary_by_category.

## Promotion / revision criteria

Promote on ≥10% metric gain with validation passing. Revise on real but sub-target
gains (keep as new base). Reject with recorded reason otherwise.