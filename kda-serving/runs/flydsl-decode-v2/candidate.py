#!/usr/bin/env python3
"""Candidate: flydsl-decode-v2 — vectorized FlyDSL KDA decode step.

v1 findings addressed:
- vec4 global loads: S read ONCE per step (v1 read it twice)
- dec kept in registers between partial staging and update (trace-time list)
- CHUNK=32 (4 chunks -> 16 barriers vs 24)
- o-reduction folded into pass 1 (S[r,col] is final after its chunk update)

Layout: one workgroup per (B,H), 128 threads = one state row each.
State [B,H,K,V] fp32 contiguous; o bf16 [B,1,H,V].
"""
import functools
import sys
from pathlib import Path

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.math as fmath
from flydsl._mlir.dialects import vector as mlir_vector
from flydsl.expr import range_constexpr
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource,
)

KK, VV = 128, 128
THREADS = KK
CHUNK = 32
NVEC = CHUNK // 4
NCHUNK = VV // CHUNK
_SCALE = KK ** -0.5


def _lane(v, j):
    """Extract static lane j of a vector value."""
    return fx.arith.ArithValue(mlir_vector.extract(v, [], [j]))


@functools.lru_cache(maxsize=4)
def _compile_decode_kernel():
    @fx.struct
    class Smem:
        # partial[K rows, CHUNK cols] + u[CHUNK], f32
        raw: fx.Array[fx.Float32, KK * CHUNK + CHUNK]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def kda_decode_step_kern(
        Q: fx.Tensor, KB: fx.Tensor, GB: fx.Tensor, Vvin: fx.Tensor,
        BETA: fx.Tensor, S_IN: fx.Tensor, S_OUT: fx.Tensor, O: fx.Tensor,
        SCALE_T: fx.Tensor,
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
        f32v4 = fx.T.VectorType.get([4], f32)

        k_f = fx.arith.extf(f32, buffer_load(k_r, row_base + tid, vec_width=1, dtype=bf16))
        g_f = fx.arith.extf(f32, buffer_load(g_r, row_base + tid, vec_width=1, dtype=bf16))
        q_f = fx.arith.extf(f32, buffer_load(q_r, row_base + tid, vec_width=1, dtype=bf16))
        beta_f = fx.arith.extf(f32, buffer_load(b_r, bh, vec_width=1, dtype=bf16))
        scale_f = buffer_load(sc_r, fx.Int32(0), vec_width=1, dtype=f32)

        # ---- L2-normalize q, k (separate norms) ----
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
        fx.rocdl.s_barrier()
        q_f = fx.arith.mulf(q_f, fx.arith.ArithValue(smem[fx.Int32(0)]))
        k_f = fx.arith.mulf(k_f, fx.arith.ArithValue(smem[fx.Int32(1)]))
        fx.rocdl.s_barrier()

        bk = fx.arith.mulf(beta_f, k_f)
        qs = fx.arith.mulf(q_f, scale_f)
        decay_v = mlir_vector.broadcast(f32v4, fmath.exp(g_f))
        bk_v = mlir_vector.broadcast(f32v4, bk)
        qs_v = mlir_vector.broadcast(f32v4, qs)

        dec_regs = [None] * NVEC  # trace-time register carry across barriers

        # ---- fused pass: per chunk, load+decay, u-reduction, update, o store ----
        for c in range_constexpr(NCHUNK):
            s_col = fx.Int32(c * CHUNK)
            # load row chunk once (vec4 x8), decay in registers
            for i in range_constexpr(NVEC):
                s_idx = s_base + tid * fx.Int32(VV) + s_col + fx.Int32(i * 4)
                dec_regs[i] = fx.arith.mulf(
                    buffer_load(s_r, s_idx, vec_width=4, dtype=f32), decay_v)
                kd = fx.arith.mulf(mlir_vector.broadcast(f32v4, k_f), dec_regs[i])
                for j in range_constexpr(4):
                    smem[tid * fx.Int32(CHUNK) + fx.Int32(i * 4 + j)] = _lane(kd, j)
            fx.rocdl.s_barrier()
            # threads 0..31: u[j] = v[j] - sum_r partial[r, j]
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
            # thread t: update state from registers (scalar lanes), stage o partial
            for i in range_constexpr(NVEC):
                for j in range_constexpr(4):
                    col = fx.Int32(i * 4 + j)
                    s_idx = s_base + tid * fx.Int32(VV) + s_col + col
                    dec_j = _lane(dec_regs[i], j)
                    u_j = fx.arith.ArithValue(
                        smem[fx.Int32(KK * CHUNK) + fx.Int32(i * 4 + j)])
                    new_s = fx.arith.addf(dec_j, fx.arith.mulf(bk, u_j))
                    buffer_store(new_s, so_r, s_idx)
                    smem[tid * fx.Int32(CHUNK) + col] = fx.arith.mulf(qs, new_s)
            fx.rocdl.s_barrier()
            # threads 0..31: o[c*32+tid] = scale * sum_r q_r * S_new[r, col]
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
            Q, KB, GB, Vvin, BETA, S_IN, S_OUT, O, SCALE_T
        ).launch(grid=(BH, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


def decode_step(q, k, g, v, beta, S, scale=_SCALE):
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
    def fwd(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
            cu_seqlens=None):
        assert q.shape[1] == 1 and q.shape[-1] == KK, "flydsl-decode-v2: T=1, K=128 only"
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
    for i in range(T):
        o, S = decode_step(q[:, i:i+1], k[:, i:i+1], g[:, i:i+1], v[:, i:i+1],
                           beta[:, i:i+1], S)
        o_ref, S_ref = kda_fp64_reference(q[:, : i + 1], k[:, : i + 1], v[:, : i + 1],
                                          g[:, : i + 1], beta[:, : i + 1])
        print(f"step {i}: o rRMSE={rel_rmse(o, o_ref[:, -1:]):.4g} "
              f"state rRMSE={rel_rmse(S, S_ref):.4g}")
    ok = rel_rmse(S, S_ref) < 0.02
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)
