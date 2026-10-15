# MI355X (gfx950) Metric Name Reference

AMD analog of `kda/skills/ncu-report-skill/reference/08-b200-metric-names.md`.

Tool: `rocprofv3` (ROCm 7.0, rocprofiler-sdk) on gfx950. Every counter name in this
doc was checked against the live counter list on this box (687 counters: 487
hardware counters in blocks SQ/SPI/TCC/TCP/TA/TD/GRBM/CPC/CPF/TCA plus 200 derived).
Values in the worked example were collected on GPU 7 on 2026-10-09.

If a counter errors or looks wrong, first check this doc, then enumerate:

```bash
rocprofv3 -L > counters.txt        # replaces --list-basic / --list-derived / --list-counters
grep -A3 "counter_name *: SQ_WAVES$" counters.txt   # description, block, dimensions
```

The listing prints every GPU in the node regardless of `HIP_VISIBLE_DEVICES`;
parse the first occurrence of each name.

---

## Names from older AMD docs that do not exist on gfx950

Older rocprof (v1) and MI100/MI200-era guides use derived metrics that `rocprofv3`
on gfx950 does not provide. Replacements:

| Older name | gfx950 / rocprofv3 replacement |
|---|---|
| `Wavefronts` | **`SQ_WAVES_sum`** |
| `VALUInsts`, `SALUInsts`, `LDSInsts` | **`SQ_INSTS_VALU`**, **`SQ_INSTS_SALU`**, **`SQ_INSTS_LDS`** (raw totals; divide by `SQ_WAVES_sum` for per-wave) |
| `FlatVMemInsts` | **`SQ_INSTS_FLAT`** |
| `L2CacheHit` | compute: `100 * TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)` |
| `KernelBusy` | **`GPU_UTIL`** (= `100 * GRBM_GUI_ACTIVE / GRBM_COUNT`) |
| `TD_BUSY` | **`TD_TD_BUSY_sum`** |
| `MemUnitBusy` | no 1:1; closest signals are `TA_BUSY_avr` / `TA_BUSY_max` relative to `GRBM_GUI_ACTIVE`, and `MemUnitStalled` |
| `WriteUnitStalled` | no 1:1; use `TA_DATA_STALLED_BY_TC_CYCLES_sum` and `TCP_TCR_TCP_STALL_CYCLES` |
| `ALUStalledByLDS` | no 1:1; use `SQ_WAIT_INST_LDS` and `LdsLatency` |
| `rocprof --list-basic / --list-derived` | **`rocprofv3 -L`** |
| `FETCH_SIZE` as "bytes read from DRAM" | **broken on gfx950: reports half.** Use `TCC_EA0_RDREQ_sum * 128` (see Gotcha 1) |
| `BANDWIDTH_EA` | **do not use on gfx950** (see Gotcha 2); compute bandwidth manually |

---

## Nsight Compute → rocprofv3 crosswalk

For readers coming from the B200 doc. "Column" means a field in rocprofv3's
`*_counter_collection.csv` or `*_kernel_trace.csv`, not a counter.

| Nsight Compute (B200) | rocprofv3 (gfx950) |
|---|---|
| `gpu__time_duration.sum` | kernel trace `End_Timestamp - Start_Timestamp` (ns) |
| `launch__grid_size` | column `Grid_Size` — **in work-items, not workgroups** (divide by `Workgroup_Size`) |
| `launch__block_size` | column `Workgroup_Size` |
| `launch__registers_per_thread` | columns `VGPR_Count`, `Accum_VGPR_Count` (AGPRs), `SGPR_Count` |
| `launch__shared_mem_per_block` | column `LDS_Block_Size` |
| local ld/st (register spill) | column `Scratch_Size` (> 0 means scratch/spill) |
| `launch__occupancy_limit_registers / _shared_mem / _warps / _blocks` | `SPI_RA_VGPR_SIMD_FULL_CSN`, `SPI_RA_LDS_CU_FULL_CSN`, `SPI_RA_WAVE_SIMD_FULL_CSN`, `SPI_RA_BAR_CU_FULL_CSN` (non-zero = that resource blocked wave launch) |
| `device__attribute_multiprocessor_count` | `CU_NUM` (256 on MI355X) |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | `OccupancyPercent`, `MeanOccupancyPerActiveCU` |
| `dram__bytes_read.sum` | `TCC_EA0_RDREQ_sum * 128` (+ `TCC_EA0_RDREQ_32B_sum * 32` if non-zero) |
| `dram__bytes_write.sum` | `WRITE_SIZE * 1024` (accurate), or from `TCC_EA0_WRREQ_sum` / `TCC_EA0_WRREQ_64B_sum` |
| `lts__t_sector_hit_rate.pct` (L2) | `TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)` |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio` | fraction of full write transactions: `TCC_EA0_WRREQ_64B_sum / TCC_EA0_WRREQ_sum` |
| `smsp__sass_inst_executed_op_global_ld / _st.sum` | `SQ_INSTS_VMEM_RD`, `SQ_INSTS_VMEM_WR` |
| `smsp__sass_inst_executed_op_shared_ld / _st.sum` | `SQ_INSTS_LDS_LOAD`, `SQ_INSTS_LDS_STORE` |
| shared-memory bank conflicts | `SQ_LDS_BANK_CONFLICT` (cycles), `LDSBankConflict` (% of GPU time) |
| `sm__inst_executed_pipe_fma` | `SQ_INSTS_VALU_FMA_F32`, `SQ_INSTS_VALU_FLOPS_FP32` |
| `sm__pipe_tensor_cycles_active` | `SQ_VALU_MFMA_BUSY_CYCLES`, `MfmaUtil` |
| `smsp__average_warps_issue_stalled_long_scoreboard_*` | no 1:1. Use `VmemLatency` with `SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES`, then PC sampling |
| `smsp__pcsamp_*` (per-PC stalls) | `rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method stochastic --pc-sampling-unit cycles` |
| `ncu --page details` "Est. Speedup" rules | `rocprof-compute` (installed; not yet wired into this skill) |

---

## Canonical gfx950 metric set (curated, verified on this box)

`_sum` / `_avr` / `_max` variants are derived reductions over hardware instances
(TCC channels, SEs, XCCs). Prefer them over the raw per-instance counters.

### Launch geometry / occupancy
```
Grid_Size, Workgroup_Size, VGPR_Count, Accum_VGPR_Count, SGPR_Count,
LDS_Block_Size, Scratch_Size                          # CSV columns, free with any --pmc run
CU_NUM  SIMD_NUM  SE_NUM  MAX_WAVE_SIZE                # device constants (256 CUs, wave64)
SQ_WAVES_sum                                           # waves launched
OccupancyPercent  MeanOccupancyPerCU  MeanOccupancyPerActiveCU
SPI_CSN_NUM_THREADGROUPS  SPI_CSN_WAVE
SPI_RA_VGPR_SIMD_FULL_CSN  SPI_RA_LDS_CU_FULL_CSN      # occupancy limiters
SPI_RA_WAVE_SIMD_FULL_CSN  SPI_RA_BAR_CU_FULL_CSN  SPI_RA_RES_STALL_CSN
```

### Timing / activity
```
GRBM_GUI_ACTIVE  GRBM_COUNT  GPU_UTIL                  # GPU-busy cycles, wall cycles, %
SQ_BUSY_CYCLES                                         # clock cycles with any wave in an SE
SQ_BUSY_CU_CYCLES                                      # quad-cycles each CU busy
SQ_WAVE_CYCLES                                         # quad-cycles waves resident
```

### DRAM / fabric traffic (read Gotcha 1 first)
```
TCC_EA0_RDREQ_sum  TCC_EA0_RDREQ_32B_sum  TCC_BUBBLE_sum   # read requests (128B on gfx950)
TCC_EA0_WRREQ_sum  TCC_EA0_WRREQ_64B_sum                   # write transactions (32B or 64B)
WRITE_SIZE                                                 # KB written (accurate)
FETCH_SIZE                                                 # KB read (REPORTS 50% on gfx950)
TCC_EA0_RDREQ_DRAM  TCC_EA0_WRREQ_DRAM                     # requests destined for HBM
TCC_EA0_RDREQ_DRAM_CREDIT_STALL_sum  TCC_EA0_WRREQ_DRAM_CREDIT_STALL_sum   # HBM back-pressure
TCC_EA0_RDREQ_GMI_CREDIT_STALL_sum                         # cross-die (GMI) back-pressure
TCC_EA0_RD_UNCACHED_32B  MemWrites32B
```

### Caches
```
TCC_HIT_sum  TCC_MISS_sum  TCC_REQ  TCC_READ  TCC_WRITE  TCC_ATOMIC   # L2
TCC_TAG_STALL  TCC_BUSY_avr
TCP_TOTAL_CACHE_ACCESSES  TCP_TCC_READ_REQ  TCP_TCC_WRITE_REQ        # vL1D <-> L2
TCP_PENDING_STALL_CYCLES  TCP_TCC_READ_REQ_LATENCY
TCP_UTCL1_TRANSLATION_HIT  TCP_UTCL1_TRANSLATION_MISS                 # TLB
SQC_DCACHE_HITS  SQC_ICACHE_HITS                                      # scalar / instruction caches
```

### Instruction mix
```
SQ_INSTS  SQ_INSTS_VALU  SQ_INSTS_SALU  SQ_INSTS_SMEM  SQ_INSTS_BRANCH
SQ_INSTS_VMEM  SQ_INSTS_VMEM_RD  SQ_INSTS_VMEM_WR  SQ_INSTS_FLAT
SQ_INSTS_LDS  SQ_INSTS_MFMA
SQ_INSTS_VALU_CVT  SQ_INSTS_VALU_INT32  SQ_INSTS_VALU_TRANS_F32
```

### VALU compute
```
SQ_INSTS_VALU_FMA_F32  SQ_INSTS_VALU_ADD_F32  SQ_INSTS_VALU_MUL_F32
SQ_INSTS_VALU_FLOPS_FP32  SQ_INSTS_VALU_FLOPS_FP16  SQ_INSTS_VALU_FLOPS_FP64  SQ_INSTS_VALU_IOPS
SQ_ACTIVE_INST_VALU  SQ_THREAD_CYCLES_VALU
VALUBusy  VALUUtilization  SALUBusy
```

### MFMA (matrix cores)
```
SQ_INSTS_VALU_MFMA_BF16  SQ_INSTS_VALU_MFMA_F16  SQ_INSTS_VALU_MFMA_F32
SQ_INSTS_VALU_MFMA_MOPS_BF16  SQ_INSTS_VALU_MFMA_MOPS_F16  SQ_INSTS_VALU_MFMA_MOPS_F32
SQ_INSTS_VALU_MFMA_MOPS_F8  SQ_INSTS_VALU_MFMA_MOPS_F6F4  SQ_INSTS_VALU_MFMA_MOPS_I8
SQ_VALU_MFMA_BUSY_CYCLES  MfmaUtil  MfmaFlops  MfmaFlopsBF16  MfmaFlopsF16  MfmaFlopsF32
```

### LDS
```
SQ_INSTS_LDS_LOAD  SQ_INSTS_LDS_STORE  SQ_INSTS_LDS_ATOMIC
SQ_INSTS_LDS_LOAD_BANDWIDTH  SQ_INSTS_LDS_STORE_BANDWIDTH
SQ_LDS_BANK_CONFLICT  SQ_LDS_ADDR_CONFLICT  SQ_LDS_UNALIGNED_STALL
SQ_LDS_DATA_FIFO_FULL  SQ_LDS_CMD_FIFO_FULL  SQ_LDS_IDX_ACTIVE
SQ_ACTIVE_INST_LDS  SQ_WAIT_INST_LDS  LDSBankConflict  LdsLatency
```

### Wave stalls and latency (the AMD substitute for stall-reason ratios)
```
SQ_WAIT_INST_ANY      # quad-cycles waves waited to issue
SQ_WAIT_ANY           # quad-cycles waves waited for anything
SQ_ACTIVE_INST_ANY  SQ_ACTIVE_INST_VMEM  SQ_ACTIVE_INST_SCA  SQ_ACTIVE_INST_MISC
SQ_INST_LEVEL_VMEM  SQ_INST_LEVEL_LDS  SQ_INST_LEVEL_SMEM     # in-flight instruction levels
VmemLatency  LdsLatency  SmemLatency  InstrFetchLatency        # avg cycles per instruction
SQ_INST_CYCLES_VMEM_RD  SQ_INST_CYCLES_VMEM_WR
```
Useful ratios (same units, so no conversion):
`SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES` = fraction of wave lifetime waiting to issue;
`SQ_ACTIVE_INST_ANY / SQ_WAVE_CYCLES` = fraction actively executing.

### Address / data path (TA, TD)
```
TA_BUSY_avr  TA_BUSY_max
TA_BUFFER_READ_WAVEFRONTS_sum  TA_BUFFER_WRITE_WAVEFRONTS_sum
TA_BUFFER_COALESCEABLE_WAVEFRONTS_sum  TA_BUFFER_COALESCED_READ_CYCLES_sum
TA_FLAT_READ_WAVEFRONTS_sum  TA_FLAT_WRITE_WAVEFRONTS_sum
TA_ADDR_STALLED_BY_TC_CYCLES_sum  TA_DATA_STALLED_BY_TC_CYCLES_sum
TD_TD_BUSY_sum  TD_TC_STALL_sum  TD_COALESCABLE_WAVEFRONT_sum
TD_LOAD_WAVEFRONT_sum  TD_STORE_WAVEFRONT_sum
MemUnitStalled
```

### PC sampling (per-instruction attribution)
gfx950 supports both methods (`rocprofv3 -L` shows `host_trap`, unit time, and
`stochastic`, unit cycle, interval power of 2 from 256):
```bash
rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method stochastic \
          --pc-sampling-unit cycles --pc-sampling-interval 1048576 \
          --kernel-include-regex "<kernel>" -d out -o run -- python3 bench.py
```
Beta in ROCm 7.0. Stochastic sampling records the stall reason of the sampled
instruction, which is the closest analog of Nsight's `--set source`.

---

## Collection recipes

One pass can only hold what the hardware can count at once. Derived metrics expand
into several TCC counters across 16 channels, so put **one derived metric per pass**:

```bash
for P in "FETCH_SIZE" "WRITE_SIZE" "TCC_HIT_sum TCC_MISS_sum" \
         "SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR" \
         "VmemLatency" "SQ_WAVE_CYCLES SQ_WAIT_INST_ANY SQ_ACTIVE_INST_ANY"; do
  tag=$(echo $P | cut -d' ' -f1)
  timeout 150 rocprofv3 --kernel-trace --pmc $P --kernel-include-regex "<kernel>" \
      -d prof/$tag -o run --output-format csv -- python3 bench.py
done
```

Alternatively list one `pmc:` line per pass in a text file and pass it with `-i`.
Take medians over at least 5 dispatches; several SQ counters are documented as
nondeterministic.

---

## Worked example: FLA decode vs our v7 kernel (Kimi K3 KDA, B=128, fp32 state)

State read and write are 805.3 MB each per step. Medians over 5 dispatches.

| Metric | FLA `fused_recurrent_kda_fwd_kernel` | v7 `kda_decode_step_kern` |
|---|---|---|
| duration | 293.8 µs | 486.2 µs |
| `Grid_Size` / `Workgroup_Size` | 12,582,912 / 256 | 12,582,912 / 256 |
| `VGPR_Count` / `SGPR_Count` / `LDS_Block_Size` / `Scratch_Size` | 44 / 64 / 0 / 0 | 32 / 48 / 0 / 0 |
| `SQ_WAVES` | 196,608 | 196,608 |
| reads: 2 × `FETCH_SIZE` (= `TCC_EA0_RDREQ_sum * 128` when no 32B reads) | 849.6 MB | 850.4 MB |
| writes: `WRITE_SIZE` | 808.5 MB | 824.7 MB |
| full 64B writes: `TCC_EA0_WRREQ_64B_sum / TCC_EA0_WRREQ_sum` | 100.0 % | 98.5 % |
| **achieved bandwidth** | **5.6 TB/s** | **3.4 TB/s** |
| L2 hit rate (`TCC_HIT_sum`, `TCC_MISS_sum`) | 32.9 % | 82.7 % |
| `SQ_INSTS_VMEM_RD` / `SQ_INSTS_VMEM_WR` per wave | 20 / 5 | 13 / 6 |
| `SQ_INSTS_VALU` per wave | 390 | 451 |
| `VmemLatency` | 2,597 cycles | 1,848 cycles |
| `SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES` | 27 % | 74 % |
| `SQ_ACTIVE_INST_ANY / SQ_WAVE_CYCLES` | 18 % | 7 % |
| total `SQ_WAVE_CYCLES` | 0.68 G | 1.93 G |

Reading: same bytes, same waves, fewer registers, full write transactions, and
lower per-op memory latency. v7 waves live 2.8× longer because they wait to issue
74 % of the time. The bottleneck is instruction dependency (the shuffle-tree
chains), not bandwidth or store layout. See `06-diagnosis-playbook.md`, case M1.

---

## Gotchas

1. **`FETCH_SIZE` reports half the bytes read on gfx950.** Its expression is
   `(TCC_BUBBLE_sum*128 + (TCC_EA0_RDREQ_sum - TCC_BUBBLE_sum - TCC_EA0_RDREQ_32B_sum)*64 + TCC_EA0_RDREQ_32B_sum*32)/1024`,
   i.e. it credits 128 bytes only to "bubble" requests. On gfx950 reads are 128-byte
   requests and `TCC_BUBBLE_sum` stays 0. Calibration: a 1 GiB `copy_` and a 1 GiB
   `sum()` both gave `FETCH_SIZE` = 0.500 GiB, `TCC_EA0_RDREQ_sum` = 8,388,792
   (1 GiB / 128 B = 8,388,608), `TCC_BUBBLE_sum` = 0; `WRITE_SIZE` = 1.000 GiB for
   the copy. Use `TCC_EA0_RDREQ_sum * 128`, and re-run the calibration after any ROCm
   upgrade.
2. **`BANDWIDTH_EA` is wrong on gfx950.** Its expression contains
   `(TCC_BUBBLE_sum - TCC_EA0_RDREQ_sum)*64`, which subtracts read traffic when
   `TCC_BUBBLE_sum` is 0. Compute bandwidth as bytes / kernel duration instead.
3. **"Request exceeds the capabilities of the hardware to collect" (error code 38)**
   means the `--pmc` set does not fit one pass (`FETCH_SIZE WRITE_SIZE GRBM_GUI_ACTIVE`
   together already fails). After this abort rocprofv3 **hangs instead of exiting**,
   so always wrap it in `timeout`.
4. **Units.** `SQ_WAVE_CYCLES`, `SQ_WAIT_INST_ANY`, `SQ_WAIT_ANY`,
   `SQ_ACTIVE_INST_ANY` and `SQ_BUSY_CU_CYCLES` are in quad-cycles (4 clocks);
   `SQ_BUSY_CYCLES` and `GRBM_GUI_ACTIVE` are in clocks. Mixing them gives ratios over
   100 %.
5. **`Grid_Size` is in work-items.** 12,582,912 with `Workgroup_Size` 256 is 49,152
   workgroups (128 batch × 96 heads × 4 column tiles).
6. **Topology.** Counters carry `DIMENSION_XCC [0:8]`: MI355X is 8 XCDs × 32 CUs, each
   XCD with its own L2. Per-instance counters cover one XCC/SE/channel; use the `_sum`
   derived forms for whole-GPU totals.
7. **Durations differ between tools.** FLA measured 293.8 µs under rocprofv3 and
   301-311 µs under torch.profiler on the same box and shapes. Compare durations only
   within one tool.
