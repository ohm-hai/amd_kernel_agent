# Task Contract — Kimi K3 KDA layer on MI355X

Layer spec (from `Primus/primus/configs/models/megatron/kimi_k3.yaml`, matching
moonshotai/Kimi-K3 config.json): KDA layers use `linear_num_key_heads=96`,
`linear_num_value_heads=96`, `linear_key_head_dim=128`, `linear_value_head_dim=128`
(H=96, K=V=128). 69 of 93 layers are KDA (rest MLA). This task optimizes the KDA
recurrence kernels on 1× AMD MI355X (gfx950). Out of scope: the layer's projections,
short convolutions, output gate and norm.

## Target

**2× FLA Triton on decode** (user target and definitions, 2026-10-09):

- **Aggregation:** geometric mean of the speedup over the serving-path grid
  B ∈ {1, 8, 32, 128} must reach 2×, and no batch size may be slower than FLA.
- **Language:** FlyDSL. v8 may continue from v7 or start from scratch.

Reference implementation for the target: FLA's **serving path**
(`fused_recurrent_kda_fwd` with `ssm_state_indices` and `inplace_final_state=True`),
which is what serving engines call and is 8–19% faster at B>=32 than the public
`fused_recurrent_kda` wrapper. Physics, from TraceLens with the MI355X spec: with fp32
state the B=128 roofline floor (203 µs) is above 2× of FLA (~146 µs), so 2× at large B
requires fp16 state (see Correctness).

## Modes

| Mode | Baseline kernel | Benchmark | Grid |
|---|---|---|---|
| Decode, serving path (official) | FLA `fused_recurrent_kda_fwd`, in-place indexed pool | `bench/serving_decode.py` | B ∈ {1, 8, 32, 128}, H=96 |
| Decode, public wrapper | FLA `fused_recurrent_kda` | `bench/matrix.py --decode-only --graph` | B ∈ {1…128} × ctx ∈ {1K, 8K, 32K} |
| Prefill | FLA `chunk_kda` | `bench/matrix.py --prefill-only` | T ∈ {1K…64K} × B ∈ {1, 2, 4, 8}, cap 160K tokens per cell |
| Chunked prefill | `chunk_kda` in 8K chunks with state carry | `bench/matrix.py` | T=64K, B=1 |

AITER has no KDA or linear-attention kernel (aiter 0.1.9.post1), so there is no AITER
baseline.

## Inputs

Use `inputs.py` with `gates="kimi"` (FLA's gate parameterization, calibrated to p99 ≈
600 bits of decay per 64 tokens, the real-model figure from the KDA^2 blog).
`gates="synthetic"` only reproduces pre-2026-10-09 runs. Timing is gate-independent
(within 1%); correctness is not.

## Correctness

Prefill candidates must pass the kda-attn hardened verifier (`../kda-attn/verifier`,
fp64 reference). Decode candidates must pass the full decode verifier
(`verifier/verify_decode.py`, 15 gates) against the pre-registered bar in
`accuracy-bar.md` **before any performance number is reported for them**.

Reduced-precision state: fp16 passes the bar (worst state rRMSE 2.0e-3); bf16 fails
(1.6% on long-memory gates). Path B uses fp16 state.

Correction (2026-10-09): earlier versions of this file said decode candidates passed
the kda-attn gates. They did not; v1–v7 were checked with a 3-step carry only. v7 has
since passed the full decode verifier (15/15).

## Official measurements

- **Wall time per step:** CUDA-graph replay, both sides graphed, serving path.
- **Kernel time:** torch.profiler or rocprofv3, the same tool for both sides; never
  compare durations across tools.
- **Efficiency:** TraceLens `linear_attention` sheet, `--gpu_arch_platform MI355X`,
  `--extension_file tracelens/kda_extension.py`.
- Medians over at least 5 steps or dispatches; warm-up excludes compile and autotune.

## Instrumentation rule

Profile candidates through `tracelens/kda_ops.py` (named ops `kda::decode_step_<impl>`)
with `record_shapes=True`, and add new implementation names to
`tracelens/kda_extension.py`. Otherwise TraceLens files the kernel under "other" and
reports no bandwidth or roofline numbers.

## Promotion criteria

A decode candidate promotes when all of these hold:

1. full decode verifier PASS (15/15) at its declared state dtype;
2. native support for the serving interface (in-place, indexed state pool) — a
   gather/scatter wrapper does not count;
3. graphed per-step latency faster than the current best on the serving path, with the
   per-B breakdown recorded. The target is met when the geomean speedup over
   B ∈ {1, 8, 32, 128} is ≥ 2× and every B is ≥ 1×.

Record every measurement with `../tools/record.py` into `results.csv`, and the candidate's
status in `candidates.jsonl`. Rows before 2026-10-09 are in `history/`.
