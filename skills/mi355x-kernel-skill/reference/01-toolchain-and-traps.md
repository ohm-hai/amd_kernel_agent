# Toolchain Idioms and Traps (measured on this box)

## Dispatch costs (wall-clock killers at small size)
- FlyDSL @flyc.jit dispatch: ~90 us/call CPU-side (measured: B=1 wall 105us - kernel 15.7us)
- Triton (FLA) dispatch: ~38 us/call
- Our Python wrapper (reshape/contiguous/empty per call): 47-133 us
- FIX: torch.cuda.CUDAGraph capture (8 alternating-state steps) -> B=1 wall 105->7 us.
  BOTH sides must be graphed for fair comparison. Serving engines graph decode anyway.

## FlyDSL (MLIR DSL; primus_turbo/flydsl has working examples)
- Plain `range()` in kernels becomes a DYNAMIC loop; use `range_constexpr` for unrolled.
- Shared memory: scalar slots only (`fx.Array[fx.Float32, N]`); vector types REJECTED.
- Vector lane access: `mlir_vector.extract(v, [], [j])`; build: `mv.insert(scalar, v, [], [j])`.
- Vector math types: extf/trunc_f need matching VECTOR dtypes (arith same-shape rule).
- Loading bf16 data: use dtype=bf16 + extf to f32 vector; NEVER dtype=f32 on bf16 tensor
  (bit-reinterpretation, silently wrong — cost us a debug cycle in v5).
- smem reads must be wrapped: `fx.arith.ArithValue(smem[idx])`.
- buffer_store(mask=scalar_predicate): FAULTS on this stack (GPU page fault) — avoid;
  use runtime branches (`if lane < N:`) or predicated static unrolls instead.
- Scalar kernel args: pass as 1-elem tensors (fx.Float32 args are fragile).
- Dispatch overhead is per-call: for loops, hoist launcher compilation
  (`_compile_decode_kernel()` is lru_cached) and pre-allocate outputs.

## TraceLens
- Installed in /opt/venv; repo at /root/amd-workspace/TraceLens.
- `TraceLens_generate_perf_report_pytorch --profile_json_path t.json
  --output_xlsx_path r.xlsx --output_csvs_dir csvs --gpu_arch_platform MI355X`
  (MI355X.json added locally 2026-10-09 with measured max-achievable FLOPs; reports made
  before that used MI300X specs and overstate roofline percentages).
- KDA ops: profile through `kda-serving/tracelens/kda_ops.py` (named ops,
  `record_shapes=True`) and pass `--extension_file kda-serving/tracelens/kda_extension.py`,
  or TraceLens files KDA kernels under "other" with no bandwidth model.
- FLA's public `fused_recurrent_kda` allocates a new state each call
  (`inplace_final_state=False`). Serving engines call `fused_recurrent_kda_fwd` with
  `ssm_state_indices` and `inplace_final_state=True`, which is 8–19% faster at B>=32
  on MI355X. Benchmark against the serving path (`kda-serving/bench/serving_decode.py`).
- Key outputs: gpu_timeline.csv (IDLE% — dispatch-bound smoking gun), unified_perf_summary
  (kernel means as synthetic ops), ops_summary_by_category.
- GPU-arch skill names above are B200 (sm_100) Nsight names; MI355X analogs via
  `rocprofv2 --pmc` or omniperf (not yet wired).

## Triton (ROCm)
- pytorch-triton-rocm 3.4.0 + torch 2.9.0a0: torch.compile/inductor BROKEN
  (KernelMetadata.cluster_dims missing) — do not use torch.compile as an optimization vehicle.
- FLA 0.5.2 kernels work (chunk_kda prefill ~3.3M tok/s; fused_recurrent decode).
- FLA Triton int32 overflow: tensors > 2^31 elements FAULT (reproduced twice) — cap
  B*T*H*K; candidate kernels must use 64-bit indexing or tile-local offsets.
- aiter 0.1.9 JIT broken (aiter<->composable_kernel mismatch): ck_tile enums missing.
