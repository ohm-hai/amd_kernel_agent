# AMD Kernel AI Agent (`kagent`) & KDA Optimization on Instinct MI355X

This repository contains the **Kernel Agent Framework (KAF / `kagent`)** and the complete research & kernel development lineage optimizing **Kimi K3 KDA (Gated Delta Attention) decode** on the **AMD Instinct MI355X (CDNA4 / `gfx950`)**.

---

## 🚀 Key Highlights & Benchmark Results

All evaluations were benchmarked on **1× AMD Instinct MI355X (288 GB HBM3E, 8.0 TB/s)** using CUDA Graph replay with an in-place indexed state pool in **pure FP32 (`float32`)**:

| Candidate | Architecture & Changes | B = 1 | B = 8 | B = 32 | B = 128 | Serving Geomean vs FLA | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FLA Triton Baseline** | Portable Triton kernel (`fused_recurrent_kda_fwd`, native pool) | 7.48 µs | 22.06 µs | 65.95 µs | 293.00 µs | **1.00×** (baseline) | Baseline |
| **`v7`** | 256-thread col-split, 0 barriers, in-wave shuffles (external gather/scatter) | 21.64 µs | 76.61 µs | 348.12 µs | 1361.72 µs | **0.25×** (wrapper-bound) | Validated |
| **`v8`** | **Native in-place indexed pool** (`decode_pool`, zero gather/scatter) | 7.28 µs | 30.96 µs | 102.66 µs | 468.07 µs | **0.74×** (+3× vs v7) | Validated |
| **`v9`** | Hoisted state loads, interleaved reduction trees, vector FMA state store | 7.13 µs | 29.35 µs | 97.45 µs | 442.50 µs | **0.77×** (+4% vs v8) | Validated |
| **`v10`** | **V-Major contiguous layout** (`state_v_first`, 512B burst transactions) | **5.67 µs** | **19.92 µs** | **65.38 µs** | **275.23 µs** | **1.12× (Beats FLA across ALL B)** | **Best Geomean** |
| **`v11`** | CU occupancy tuning: 128-thread blocks, NV=8 (768 blocks at B=1) | **5.43 µs** | **20.05 µs** | 69.98 µs | 279.07 µs | **1.11× (Record B=1 latency: 1.38×)** | Validated |
| **`v12`** | Adaptive dispatch: NV=8 for B<=8, NV=4 for B>=32 | **5.43 µs** | **20.08 µs** | 73.61 µs | 277.44 µs | **1.09×** | Validated |

### Breakthroughs in `v10` - `v12`:
1. **100% Coalesced Burst Memory Transactions (V-Major Layout):**
   - Transposing recurrent state from $[K, V]$ to $[V, K]$ (`state_v_first`) ensures all $K=128$ rows are contiguous in memory.
   - For any column, the 64 lanes of a Wave64 read 512 contiguous bytes in **exactly four 128-byte burst transactions** at 100% bus utilization.
   - At $B=128$, latency dropped from 442 µs to **275.2 µs**, reaching **74.0% of peak theoretical HBM3E bandwidth (5.92 TB/s)**.
2. **Zero-LDS, Zero-Barrier Intra-Wavefront Reductions:**
   - 1 Wave64 wavefront contains 64 lanes; each lane owns 2 rows $\implies 64 \times 2 = 128$ rows.
   - The entire $K=128$ reduction dimension fits inside a single wavefront using in-wave butterfly XOR shuffles (`ds_swizzle_b32` / `v_permlane`) with **zero LDS** and **zero synchronization barriers**.
3. **Adaptive CU Wave Residency Tuning:**
   - At $B \le 8$, re-partitioning to 128-thread blocks ($NV=8$) doubles the grid at $B=1$ from 384 to **768 workgroups** (exactly 3 workgroups / 6 Wave64s per CU across all 256 CUs), eliminating SIMD starvation during memory wait states and achieving a record **5.43 µs latency (1.38× faster than FLA)**.

---

## 🛡️ The Kernel Agent Framework (`kagent` / KAF)

`kagent` (`framework/`) is a driver-agnostic, tamper-proof agent pipeline ensuring that AI kernel generation is bounded by formal mathematical proofs, frozen evaluations, and machine telemetry:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          kagent Loop Architecture                      │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ 1. Formal Verification (kagent.formal - Z3 Prover)     │
       │    - MLIR DAG parser, exact-real KDA specification      │
       │    - 19 butterfly XOR reduction chains PROVED           │
       └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ 2. Frozen Evaluator (kagent.eval)                       │
       │    - SHA-256 hash-locked crate with 15 numerical gates  │
       │    - 10/10 secret salted holdout gates (counts-only)    │
       │    - Automatic tamper refusal self-test                 │
       └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ 3. Benchmark & Telemetry (kagent.evidence)              │
       │    - CUDA Graph replay benchmarking                     │
       │    - ISA Analyzer: ALU-to-data instruction ratios       │
       │    - PC Sampling: rocprofv3 cycle stall attribution     │
       │    - Roofline: effective vs peak HBM bandwidth          │
       └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ 4. Independent Reviewer Subagent                        │
       │    - Automated audit across 8 integrity checkpoints     │
       │    - Generates formal review.md for every candidate     │
       └─────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
.
├── framework/                 # Kernel Agent Framework (kagent)
│   ├── kagent/
│   │   ├── eval/              # Hash-locked grader crate & holdout runner
│   │   ├── formal/            # MLIR parser, KDA Z3 spec, butterfly checks, lints
│   │   ├── evidence/          # ISA analyzer, rocprofv3 PC sampling, roofline
│   │   └── loop/              # Stage machine & kagent CLI
│   ├── docs/guidelines/       # Methodology & anti-reward-hacking guidelines
│   └── examples/kda-serving/  # Reference task pack with declarative cases (*.toml)
├── kda-serving/               # Kimi K3 KDA Decode Campaign
│   ├── runs/                  # Candidate lineage (v1 -> v12) with full artifacts
│   │   ├── flydsl-decode-v10/ # Best pure FP32 geomean (1.12x vs FLA)
│   │   ├── flydsl-decode-v11/ # Best low-latency (5.43 µs, 1.38x vs FLA)
│   │   └── flydsl-decode-v12/ # Adaptive dispatch candidate
│   ├── bench/                 # Serving-interface benchmarks (in-place indexed pool)
│   ├── verifier/              # 15-gate hardened numerical verifier
│   ├── accuracy-bar.md        # Pre-registered error thresholds
│   ├── candidates.jsonl       # Historical candidate ledger
│   └── results.csv            # Structured performance measurements
├── skills/
│   ├── KernelWiki/            # AMD CDNA4 & GFX950 architectural knowledge base
│   ├── mi355x-kernel-skill/   # Architecture reference, diagnosis playbook & counters
│   └── agent-loop/            # Cursor agent loop integration skill
├── tools/                     # Shared environment stamping, recording, launch floor tools
├── env.sh                     # Environment setup script
└── AGENTS.md                  # Working agreements, guidelines, and toolchain rules
```

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
source env.sh
export HIP_VISIBLE_DEVICES=0   # Set to target MI355X GPU
```

### 2. Run kagent Verification & Evaluation
Validate task pack:
```bash
python3 -m kagent.loop.cli check --pack framework/examples/kda-serving
```

Run full end-to-end pipeline on candidate `v10`:
```bash
cd framework
python3 -m kagent.loop.cli run \
    --pack examples/kda-serving \
    --candidate ../kda-serving/runs/flydsl-decode-v10 \
    --stages verify,holdout,formal,bench,evidence,report \
    --dtype float32
```

---

## 📜 License
Apache-2.0
