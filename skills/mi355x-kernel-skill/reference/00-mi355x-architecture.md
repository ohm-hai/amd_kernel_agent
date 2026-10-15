# gfx950 / MI355X Architecture Reference (verified on-box unless noted)

rocminfo (this box, 8x MI355X): gfx950:sramecc+:xnack-, 256 CU, wavefront 64,
max clock 2400 MHz, L1 32KB/CU, L2 4MB (reported per XCD), L3 262144 KB
(256 MB Infinity Cache), 288 GB HBM3E (spec: ~8 TB/s).

## Execution model (CDNA4)
- 256 CUs = 8 XCDs x 32 CUs. rocprofv3 counters carry `DIMENSION_XCC [0:8]`.
- CU = 4 SIMDs. Each SIMD keeps up to 8 waves resident (32 per CU, the divisor in
  rocprofv3's `OccupancyPercent`) and issues from one of them per cycle.
- Register pressure only reduces occupancy when per-wave VGPR allocation is large.
  Our v7 kernel uses 32 VGPRs and FLA's decode kernel 44 (`VGPR_Count`, measured);
  check `SPI_RA_VGPR_SIMD_FULL_CSN` before assuming registers limit occupancy.
- No warp-specialization culture like Hopper/Blackwell; no TMA. Async data movement via
  buffer loads w/ cache hints (evict_first etc.), multi-wave software pipelining.

## Memory hierarchy
- Per lane: scalar (SGPR) + vector (VGPR) registers; bf16/fp32 values.
- LDS 160 KB per CU, all of it allocatable by one workgroup (HIP `sharedMemPerBlock` =
  163840; rocprofv3 `Lds_Size_In_Kb` = 160). Older docs say 64 KB; that was CDNA3.
  `shuffle_xor` (ds_swizzle / ds_bpermute) exchanges lanes inside a
  wavefront and is exact for xor trees (probe in kda-serving). It goes through the LDS
  pipe, so long chains of dependent swizzles stall issue: v7's 19 trees x 6 steps
  coincide with 74% of wave lifetime spent waiting to issue (see 08 worked example).
  Confirm per instruction with PC sampling before restructuring.
- L2 -> fabric traffic, measured with rocprofv3: reads are 128 B requests
  (`TCC_EA0_RDREQ`), writes are 64 B or 32 B transactions (`TCC_EA0_WRREQ_64B`). v7
  achieves 98.5% full 64 B writes and FLA 100%, so store layout is not a bottleneck at
  this shape (the v3 lesson, now backed by counters).
- 4 MB L2 per XCD, not shared across XCDs; 256 MB Infinity Cache shared by all XCDs.
  Redundant same-address loads across lanes are served from vL1D/L2.

## MFMA (matrix) units
- bf16 mfma_16x16x16 = 4096 FLOP/inst vs VALU 64-lane FMA = 128 FLOP/cycle.
- For GEMV (M=1..2): MFMA pads M to 16 -> ~12.5% efficiency -> ~0.3x of VALU. AVOID.
- MFMA shines at real GEMM shapes (kda PREFILL if we ever hand-tune WY-representation).

## Numerics relevant to KDA/linear attention
- State accumulator precision: fp32 state = FLA convention (accurate ~0.3% @ 8k);
  bf16 state = FlashKDA (Moonshot) choice: 3.5% @ 8k (KDA^2 blog finding "hack-E analog").
  fp16 state: same 2B traffic, 10 mantissa bits vs bf16 7 -> strictly better precision;
  range risk must be measured (planned v8).
- exp/decay: fine in fp32 VALU; no fp8 math needed for decode path.
