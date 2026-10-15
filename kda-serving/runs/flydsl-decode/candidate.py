#!/usr/bin/env python3
"""Candidate: flydsl-decode — KDA decode step kernel written in FlyDSL (MLIR/ROCm).

One FlyDSL workgroup per (batch, head). 128 threads, thread t owns state row
k=t (K=V=128, Kimi K3 KDA layer shape). Per token step, fused in one kernel:

    q, k <- L2-normalize(q, k)               (block reductions, KDAttn spec)
    S <- S * exp(g)                          (per-channel decay, registers)
    kv  = k . S_decay                        (shared-mem reduction, per chunk)
    u   = v - kv                             (delta error)
    S <- S_decay + beta * k (outer) u        (delta-rule update, stored)
    o   = scale * q . S                      (second shared-mem reduction)

Replaces fla.ops.kda.fused_recurrent_kda (Triton) for T=1 decode steps.
State layout identical: [B, H, K, V] fp32 contiguous; o bf16 [B, 1, H, V].
"""
import functools
import os
import sys
from pathlib import Path

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.math as fmath
from flydsl.expr import range_constexpr
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
)

KK, VV = 128, 128          # Kimi K3 KDA head dims
THREADS = KK               # one thread per state row
CHUNK = 16                 # V processed in 8 chunks of 16
NCHUNK = VV // CHUNK
_SCALE = KK ** -0.5
DEBUG = bool(int(os.environ.get("KDA_DEBUG", 0)))


@functools.lru_cache(maxsize=4)
def _compile_decode_kernel(debug=DEBUG):
    """Compile the FlyDSL decode-step kernel; returns the @flyc.jit launcher."""

    @fx.struct
    class Smem:
        # partial[K rows, CHUNK cols] + u[CHUNK] + 2 norm slots, f32
        raw: fx.Array[fx.Float32, KK * CHUNK + CHUNK + 2]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def kda_decode_step_kern(
        Q: fx.Tensor,      # [B*H, K]   bf16
        KB: fx.Tensor,     # [B*H, K]   bf16
        GB: fx.Tensor,     # [B*H, K]   bf16 (log-decay, <= 0)
        Vvin: fx.Tensor,   # [B*H, V]   bf16
        BETA: fx.Tensor,   # [B*H]      bf16 (per-head scalar)
        S_IN: fx.Tensor,   # [B*H, K, V] fp32
        S_OUT: fx.Tensor,  # [B*H, K, V] fp32
        O: fx.Tensor,      # [B*H, V]   bf16
        SCALE_T: fx.Tensor,  # [1] f32
        dbg: fx.Constexpr[bool],
    ):
        tid = fx.thread_idx.x
        bh = fx.block_idx.x

        q_r = create_buffer_resource(Q, max_size=True)
        k_r = create_buffer_resource(KB, max_size=True)
        g_r = create_buffer_resource(GB, max_size=True)
        v_r = create_buffer_resource(Vvin, max_size=True)
        b_r = create_buffer_resource(BETA, max_size=True)
        s_r = create_buffer_resource(S_IN, max_size=True)
        so_r = create_buffer_resource(S_OUT, max_size=True)
        o_r = create_buffer_resource(O, max_size=True)
        sc_r = create_buffer_resource(SCALE_T, max_size=True)

        smem = fx.SharedAllocator().allocate(Smem).peek().raw

        row_base = bh * fx.Int32(KK)
        s_base = bh * fx.Int32(KK * VV)
        f32 = fx.T.f32()
        bf16 = fx.T.bf16()

        k_f = fx.arith.extf(f32, buffer_load(k_r, row_base + tid, vec_width=1, dtype=bf16))
        g_f = fx.arith.extf(f32, buffer_load(g_r, row_base + tid, vec_width=1, dtype=bf16))
        q_f = fx.arith.extf(f32, buffer_load(q_r, row_base + tid, vec_width=1, dtype=bf16))
        beta_f = fx.arith.extf(f32, buffer_load(b_r, bh, vec_width=1, dtype=bf16))
        scale_f = buffer_load(sc_r, fx.Int32(0), vec_width=1, dtype=f32)

        # ---- L2-normalize q and k (separate norms; delta rule needs the
        # unit sphere). Per-thread squares staged in smem, thread 0 reduces. ----
        smem[tid] = fx.arith.mulf(q_f, q_f)
        smem[fx.Int32(KK) + tid] = fx.arith.mulf(k_f, k_f)
        fx.rocdl.s_barrier()
        if tid == fx.Int32(0):
            sq = fx.arith.ArithValue(smem[fx.Int32(0)])
            sk = fx.arith.ArithValue(smem[fx.Int32(KK)])
            for r in range_constexpr(1, KK):
                e = fx.Int32(r)
                sq = fx.arith.addf(sq, fx.arith.ArithValue(smem[e]))
                sk = fx.arith.addf(sk, fx.arith.ArithValue(smem[fx.Int32(KK) + e]))
            smem[fx.Int32(0)] = fmath.rsqrt(sq)
            smem[fx.Int32(1)] = fmath.rsqrt(sk)
            if dbg:  # debug: dump raw sums into fp32 state scratch
                buffer_store(sq, so_r, s_base)
                buffer_store(sk, so_r, s_base + fx.Int32(1))
        fx.rocdl.s_barrier()
        q_f = fx.arith.mulf(q_f, fx.arith.ArithValue(smem[fx.Int32(0)]))
        k_f = fx.arith.mulf(k_f, fx.arith.ArithValue(smem[fx.Int32(1)]))
        fx.rocdl.s_barrier()  # norms read by all threads before smem reuse

        if not dbg:
            beta_k = fx.arith.mulf(beta_f, k_f)
            decay = fmath.exp(g_f)

            # ---- pass 1: decay + delta-rule update, chunk by chunk ----
            for c in range_constexpr(NCHUNK):
                s_col = fx.Int32(c * CHUNK)
                for j in range_constexpr(CHUNK):
                    s_idx = s_base + tid * fx.Int32(VV) + s_col + fx.Int32(j)
                    dec = fx.arith.mulf(
                        buffer_load(s_r, s_idx, vec_width=1, dtype=f32), decay)
                    smem[tid * fx.Int32(CHUNK) + fx.Int32(j)] = fx.arith.mulf(k_f, dec)
                fx.rocdl.s_barrier()
                if tid < fx.Int32(CHUNK):
                    acc = fx.arith.extf(
                        f32,
                        buffer_load(v_r, bh * fx.Int32(VV) + s_col + fx.Int32(tid),
                                    vec_width=1, dtype=bf16))
                    for r in range_constexpr(KK):
                        acc = fx.arith.subf(
                            acc,
                            fx.arith.ArithValue(
                                smem[fx.Int32(r) * fx.Int32(CHUNK) + fx.Int32(tid)]))
                    smem[fx.Int32(KK * CHUNK) + fx.Int32(tid)] = acc
                fx.rocdl.s_barrier()
                for j in range_constexpr(CHUNK):
                    s_idx = s_base + tid * fx.Int32(VV) + s_col + fx.Int32(j)
                    dec = fx.arith.mulf(
                        buffer_load(s_r, s_idx, vec_width=1, dtype=f32), decay)
                    u_j = fx.arith.ArithValue(
                        smem[fx.Int32(KK * CHUNK) + fx.Int32(j)])
                    buffer_store(fx.arith.addf(dec, fx.arith.mulf(beta_k, u_j)),
                                 so_r, s_idx)
                fx.rocdl.s_barrier()

            # ---- pass 2: o = scale * q . S_out (shared-mem reduction) ----
            for c in range_constexpr(NCHUNK):
                s_col = fx.Int32(c * CHUNK)
                for j in range_constexpr(CHUNK):
                    s_idx = s_base + tid * fx.Int32(VV) + s_col + fx.Int32(j)
                    p = fx.arith.mulf(
                        fx.arith.mulf(q_f, scale_f),
                        buffer_load(so_r, s_idx, vec_width=1, dtype=f32))
                    smem[tid * fx.Int32(CHUNK) + fx.Int32(j)] = p
                fx.rocdl.s_barrier()
                if tid < fx.Int32(CHUNK):
                    acc = fx.arith.ArithValue(smem[fx.Int32(tid)])
                    for r in range_constexpr(1, KK):
                        acc = fx.arith.addf(
                            acc,
                            fx.arith.ArithValue(
                                smem[fx.Int32(r) * fx.Int32(CHUNK) + fx.Int32(tid)]))
                    buffer_store(
                        fx.arith.trunc_f(bf16, acc),
                        o_r, bh * fx.Int32(VV) + s_col + fx.Int32(tid))
                fx.rocdl.s_barrier()

    @flyc.jit
    def launch(
        Q: fx.Tensor, KB: fx.Tensor, GB: fx.Tensor, Vvin: fx.Tensor,
        BETA: fx.Tensor, S_IN: fx.Tensor, S_OUT: fx.Tensor, O: fx.Tensor,
        SCALE_T: fx.Tensor, BH: fx.Int32, stream: fx.Stream,
    ):
        kda_decode_step_kern(
            Q, KB, GB, Vvin, BETA, S_IN, S_OUT, O, SCALE_T, DEBUG
        ).launch(grid=(BH, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


def decode_step(q, k, g, v, beta, S, scale=_SCALE):
    """One KDA decode step. q,k,g,v: [B,1,H,*] bf16; beta: [B,1,H] bf16;
    S: [B,H,K,V] fp32 -> returns (o [B,1,H,V] bf16, S_new fp32)."""
    B, _, H, _ = q.shape
    BH = B * H
    launch = _compile_decode_kernel()
    s_out = torch.empty_like(S)
    o = torch.empty((B, 1, H, VV), dtype=torch.bfloat16, device=q.device)
    scale_t = torch.tensor([float(scale)], dtype=torch.float32, device=q.device)
    launch(
        q.reshape(BH, KK).contiguous(), k.reshape(BH, KK).contiguous(),
        g.reshape(BH, KK).contiguous(), v.reshape(BH, VV).contiguous(),
        beta.reshape(BH).contiguous(),
        S.contiguous(), s_out, o, scale_t, BH, torch.cuda.current_stream(),
    )
    return o, s_out


def get_forward():
    """Matrix-bench interface (matches fla fused_recurrent decode path)."""
    def fwd(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
            cu_seqlens=None):
        assert q.shape[1] == 1 and q.shape[-1] == KK, "flydsl-decode: T=1, K=128 only"
        if scale is None:
            scale = _SCALE
        S_in = (initial_state if initial_state is not None else
                torch.zeros(q.shape[0], q.shape[2], KK, VV,
                            dtype=torch.float32, device=q.device))
        o, S_out = decode_step(q, k, g, v, beta, S_in, scale=scale)
        fwd.last_final_state = S_out if output_final_state else None
        return o

    fwd.supports_final_state = True
    return fwd


if __name__ == "__main__":
    # standalone self-test: 3-step carry vs fp64 reference
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "kda-attn"))
    from ref_fp64 import kda_fp64_reference, rel_rmse

    torch.manual_seed(0)
    B, H, T = 2, 8, 3
    dev = "cuda"
    q = torch.randn(B, T, H, KK, device=dev).bfloat16()
    k = torch.randn(B, T, H, KK, device=dev).bfloat16()
    v = torch.randn(B, T, H, VV, device=dev).bfloat16()
    g = -torch.rand(B, T, H, KK, device=dev).bfloat16().abs()
    beta = torch.rand(B, T, H, device=dev).bfloat16().sigmoid()
    S = torch.zeros(B, H, KK, VV, dtype=torch.float32, device=dev)
    for i in range(3):
        o, S = decode_step(q[:, i:i+1], k[:, i:i+1], g[:, i:i+1], v[:, i:i+1],
                           beta[:, i:i+1], S)
        o_ref, S_ref = kda_fp64_reference(q[:, : i + 1], k[:, : i + 1], v[:, : i + 1],
                                          g[:, : i + 1], beta[:, : i + 1])
        print(f"step {i}: o rRMSE={rel_rmse(o, o_ref[:, -1:]):.4g} "
              f"state rRMSE={rel_rmse(S, S_ref):.4g}")
    ok = rel_rmse(S, S_ref) < 0.02
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)
