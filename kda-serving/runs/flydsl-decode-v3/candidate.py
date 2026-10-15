#!/usr/bin/env python3
"""Candidate: flydsl-decode-v3 — coalesced stores + fused-o FlyDSL KDA decode step.

v2 measured bottlenecks -> v3 changes:
1. o-reduction eliminated via algebra: with qk = q_hat . k_hat (per head),
   o[j] = od[j] + scale * beta * qk * u[j], where od = scale * q_hat . S_decay.
   od is staged/reduced in the SAME smem pass as kv -> no second pass, no
   S_out read-back. (12 barriers vs 16; pass 2 gone.)
2. Coalesced state stores: new_s is staged in smem in GLOBAL-ADDRESS order
   (m = row*32 + col_in_chunk), then thread t writes the 32 consecutive
   elements m = t*32 .. t*32+31 -> 4 threads cover one row contiguously
   (v2's thread-owns-row stores were stride-128 floats = 8x wasted sectors).

Layout identical: state [B,H,K,V] fp32 contiguous; o bf16 [B,1,H,V].
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

# smem layout (floats): kv_partial[4096] | od_partial[4096] | u[32] | qk_part[128] | norms[2]
OFF_OD = KK * CHUNK
OFF_U = 2 * KK * CHUNK
OFF_QK = OFF_U + CHUNK
OFF_NM = OFF_QK + KK
SMEM_LEN = OFF_NM + 2


def _lane(v, j):
    return fx.arith.ArithValue(mlir_vector.extract(v, [], [j]))


@functools.lru_cache(maxsize=4)
def _compile_decode_kernel():
    @fx.struct
    class Smem:
        raw: fx.Array[fx.Float32, SMEM_LEN]

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

        # ---- L2-normalize q, k (separate norms) + qk dot ----
        smem[tid] = fx.arith.mulf(q_f, q_f)
        smem[fx.Int32(KK) + tid] = fx.arith.mulf(k_f, k_f)
        smem[fx.Int32(OFF_QK) + tid] = fx.arith.mulf(q_f, k_f)
        fx.rocdl.s_barrier()
        if tid == fx.Int32(0):
            sq = fx.arith.ArithValue(smem[fx.Int32(0)])
            sk = fx.arith.ArithValue(smem[fx.Int32(KK)])
            qk = fx.arith.ArithValue(smem[fx.Int32(OFF_QK)])
            for r in range_constexpr(1, KK):
                e = fx.Int32(r)
                sq = fx.arith.addf(sq, fx.arith.ArithValue(smem[e]))
                sk = fx.arith.addf(sk, fx.arith.ArithValue(smem[fx.Int32(KK) + e]))
                qk = fx.arith.addf(qk, fx.arith.ArithValue(smem[fx.Int32(OFF_QK) + e]))
            smem[fx.Int32(0)] = fmath.rsqrt(sq)
            smem[fx.Int32(1)] = fmath.rsqrt(sk)
            smem[fx.Int32(OFF_NM)] = qk
        fx.rocdl.s_barrier()
        q_f = fx.arith.mulf(q_f, fx.arith.ArithValue(smem[fx.Int32(0)]))
        k_f = fx.arith.mulf(k_f, fx.arith.ArithValue(smem[fx.Int32(1)]))
        fx.rocdl.s_barrier()

        # constants re-derived post-normalization
        smem[fx.Int32(OFF_QK) + tid] = fx.arith.mulf(q_f, k_f)
        fx.rocdl.s_barrier()
        if tid == fx.Int32(0):
            qk = fx.arith.ArithValue(smem[fx.Int32(OFF_QK)])
            for r in range_constexpr(1, KK):
                qk = fx.arith.addf(
                    qk, fx.arith.ArithValue(smem[fx.Int32(OFF_QK) + fx.Int32(r)]))
            smem[fx.Int32(OFF_NM)] = qk
        fx.rocdl.s_barrier()
        qk_n = fx.arith.ArithValue(smem[fx.Int32(OFF_NM)])

        bk = fx.arith.mulf(beta_f, k_f)
        qs = fx.arith.mulf(q_f, scale_f)
        decay_v = mlir_vector.broadcast(f32v4, fmath.exp(g_f))
        bk_v = mlir_vector.broadcast(f32v4, bk)
        qs_v = mlir_vector.broadcast(f32v4, qs)
        # o = od + scale * beta * qk * u ; scale folded into od partials
        sbq_v = mlir_vector.broadcast(f32v4, fx.arith.mulf(scale_f, beta_f))

        dec_regs = [None] * NVEC

        for c in range_constexpr(NCHUNK):
            s_col = fx.Int32(c * CHUNK)
            # load row chunk once (vec4 x8), decay in registers
            for i in range_constexpr(NVEC):
                s_idx = s_base + tid * fx.Int32(VV) + s_col + fx.Int32(i * 4)
                dec_regs[i] = fx.arith.mulf(
                    buffer_load(s_r, s_idx, vec_width=4, dtype=f32), decay_v)
                kd = fx.arith.mulf(mlir_vector.broadcast(f32v4, k_f), dec_regs[i])
                od = fx.arith.mulf(qs_v, dec_regs[i])
                for j in range_constexpr(4):
                    smem[tid * fx.Int32(CHUNK) + fx.Int32(i * 4 + j)] = _lane(kd, j)
                    smem[fx.Int32(OFF_OD) + tid * fx.Int32(CHUNK)
                         + fx.Int32(i * 4 + j)] = _lane(od, j)
            fx.rocdl.s_barrier()
            # threads 0..31: u, o (od reduced alongside kv) -> store o, save u
            if tid < fx.Int32(CHUNK):
                v_j = fx.arith.extf(
                    f32,
                    buffer_load(v_r, bh * fx.Int32(VV) + s_col + fx.Int32(tid),
                                vec_width=1, dtype=bf16))
                kv = fx.arith.ArithValue(smem[fx.Int32(tid)])
                od = fx.arith.ArithValue(smem[fx.Int32(OFF_OD) + fx.Int32(tid)])
                for r in range_constexpr(1, KK):
                    e = fx.Int32(r)
                    kv = fx.arith.addf(
                        kv, fx.arith.ArithValue(
                            smem[e * fx.Int32(CHUNK) + fx.Int32(tid)]))
                    od = fx.arith.addf(
                        od, fx.arith.ArithValue(
                            smem[fx.Int32(OFF_OD) + e * fx.Int32(CHUNK) + fx.Int32(tid)]))
                u_j = fx.arith.subf(v_j, kv)
                # o = od + scale * beta * qk * u  (od already has scale folded)
                o_j = fx.arith.addf(
                    od,
                    fx.arith.mulf(
                        fx.arith.mulf(scale_f, beta_f),
                        fx.arith.mulf(qk_n, u_j)))
                buffer_store(fx.arith.trunc_f(bf16, o_j),
                             o_r, bh * fx.Int32(VV) + s_col + fx.Int32(tid))
                smem[fx.Int32(OFF_U) + fx.Int32(tid)] = u_j
            fx.rocdl.s_barrier()
            # thread t: new_s = dec + beta*k*u (u from smem) -> stage in
            # global-address order (m = row*32 + col_in_chunk)
            for i in range_constexpr(NVEC):
                for j in range_constexpr(4):
                    dec_j = _lane(dec_regs[i], j)
                    u_j = fx.arith.ArithValue(
                        smem[fx.Int32(OFF_U) + fx.Int32(i * 4 + j)])
                    new_s = fx.arith.addf(dec_j, fx.arith.mulf(bk, u_j))
                    smem[tid * fx.Int32(CHUNK) + fx.Int32(i * 4 + j)] = new_s
            fx.rocdl.s_barrier()
            # coalesced store: thread t writes 32 consecutive global elements
            # m = t*32 .. t*32+31  (4 threads cover one row contiguously)
            m0 = tid * fx.Int32(CHUNK)
            for k in range_constexpr(CHUNK):
                m = m0 + fx.Int32(k)
                r = m // fx.Int32(CHUNK)
                jl = m - r * fx.Int32(CHUNK)
                buffer_store(
                    fx.arith.ArithValue(smem[m]),
                    so_r, s_base + r * fx.Int32(VV) + s_col + jl)
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
        assert q.shape[1] == 1 and q.shape[-1] == KK, "flydsl-decode-v3: T=1, K=128 only"
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
