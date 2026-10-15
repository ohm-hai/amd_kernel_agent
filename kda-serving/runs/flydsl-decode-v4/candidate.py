#!/usr/bin/env python3
"""Candidate: flydsl-decode-v4 — FLA-structure port: column-tile programs.

FLA source analysis (fused_recurrent.py): grid = (B*HV)*NV programs with
BK=128 (whole K per program), BV=32; state tile in registers; kv/o reduced
locally; 4 col-tiles of a head cover each state row contiguously.

v4 = that structure in FlyDSL:
- grid (B*H*NV), NV=4; 128 threads, thread t = tile row t (32 cols, 8 vec4)
- L2N (q,k) + qk via 3-phase staged reduction (3 barriers)
- main step: decay+stage kv/od partials (barrier), 32-thread column reduction
  (pipelined 128-deep FADD chains), u/o + register state update (barrier)
- fused-o: o = od + scale*beta*qk*u, od = scale*q_hat.S_decay (no pass 2)
- 6 barriers/token vs v2's 16; state stores straight from registers
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
NV = 4
BV = VV // NV          # 32 cols per program
NVEC = BV // 4         # 8 vec4s per row
_SCALE = KK ** -0.5

# smem float slots
OFF_KVP = 0                   # kv partial [row t][col j] = 4096
OFF_ODP = KK * BV             # od partial = 4096
OFF_SQ = 2 * KK * BV          # q^2 partials [128]
OFF_SK = OFF_SQ + KK          # k^2 partials [128]
OFF_QK = OFF_SK + KK          # q*k partials [128]
OFF_R1 = OFF_QK + KK          # 8-lane reduced partials (sq|sk|qk): 24
OFF_KVR = OFF_R1 + 24         # kv reduced [32]
OFF_ODR = OFF_KVR + BV        # od reduced [32]
OFF_U = OFF_ODR + BV          # u [32]
OFF_NM = OFF_U + BV           # inv_q, inv_k, qk [3]
SMEM_LEN = OFF_NM + 3


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
        iv = bh % fx.Int32(NV)
        nh = bh // fx.Int32(NV)

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

        f32 = fx.T.f32()
        bf16 = fx.T.bf16()
        f32v4 = fx.T.VectorType.get([4], f32)

        col0 = iv * fx.Int32(BV)
        row_base = nh * fx.Int32(KK)
        s_head = nh * fx.Int32(KK * VV)
        v_base = nh * fx.Int32(VV)

        k_f = fx.arith.extf(f32, buffer_load(k_r, row_base + tid, vec_width=1, dtype=bf16))
        g_f = fx.arith.extf(f32, buffer_load(g_r, row_base + tid, vec_width=1, dtype=bf16))
        q_f = fx.arith.extf(f32, buffer_load(q_r, row_base + tid, vec_width=1, dtype=bf16))
        beta_f = fx.arith.extf(f32, buffer_load(b_r, nh, vec_width=1, dtype=bf16))
        scale_f = buffer_load(sc_r, fx.Int32(0), vec_width=1, dtype=f32)

        # ---------- L2N phase 1: stage partials ----------
        smem[fx.Int32(OFF_SQ) + tid] = fx.arith.mulf(q_f, q_f)
        smem[fx.Int32(OFF_SK) + tid] = fx.arith.mulf(k_f, k_f)
        smem[fx.Int32(OFF_QK) + tid] = fx.arith.mulf(q_f, k_f)
        fx.rocdl.s_barrier()

        # ---------- L2N phase 2: 24 threads reduce 16 each ----------
        # threads 0..7 -> sq 8 partials; 8..15 -> sk; 16..23 -> qk
        g8 = tid % fx.Int32(8)
        grp = tid // fx.Int32(8)          # 0,1,2 -> which reduction
        if tid < fx.Int32(24):
            base = fx.Int32(OFF_SQ) if grp == fx.Int32(0) else (
                fx.Int32(OFF_SK) if grp == fx.Int32(1) else fx.Int32(OFF_QK))
            acc = fx.arith.ArithValue(smem[base + fx.Int32(g8)])
            for r in range_constexpr(1, 16):
                e = fx.Int32(g8 + r * 8)
                acc = fx.arith.addf(acc, fx.arith.ArithValue(smem[base + e]))
            smem[fx.Int32(OFF_R1) + tid] = acc
        fx.rocdl.s_barrier()

        # ---------- L2N phase 3: threads 0..7 finalize ----------
        if tid < fx.Int32(8):
            sq = fx.arith.ArithValue(smem[fx.Int32(OFF_R1) + fx.Int32(tid % 8)]) if False else \
                fx.arith.ArithValue(smem[fx.Int32(OFF_R1) + fx.Int32(0)])
            sq = fx.arith.ArithValue(smem[fx.Int32(OFF_R1)])
            sk = fx.arith.ArithValue(smem[fx.Int32(OFF_R1) + fx.Int32(8)])
            qk = fx.arith.ArithValue(smem[fx.Int32(OFF_R1) + fx.Int32(16)])
            for r in range_constexpr(1, 8):
                sq = fx.arith.addf(sq, fx.arith.ArithValue(smem[fx.Int32(OFF_R1) + fx.Int32(r)]))
                sk = fx.arith.addf(sk, fx.arith.ArithValue(smem[fx.Int32(OFF_R1) + fx.Int32(8 + r)]))
                qk = fx.arith.addf(qk, fx.arith.ArithValue(smem[fx.Int32(OFF_R1) + fx.Int32(16 + r)]))
            inv_q = fmath.rsqrt(sq)
            inv_k = fmath.rsqrt(sk)
            smem[fx.Int32(OFF_NM)] = inv_q
            smem[fx.Int32(OFF_NM) + fx.Int32(1)] = inv_k
            # qk of NORMALIZED vectors: sum(q_raw*k_raw) / (|q|*|k|)
            smem[fx.Int32(OFF_NM) + fx.Int32(2)] = fx.arith.mulf(
                fx.arith.mulf(qk, inv_q), inv_k)
        fx.rocdl.s_barrier()
        q_f = fx.arith.mulf(q_f, fx.arith.ArithValue(smem[fx.Int32(OFF_NM)]))
        k_f = fx.arith.mulf(k_f, fx.arith.ArithValue(smem[fx.Int32(OFF_NM) + fx.Int32(1)]))
        qk_n = fx.arith.ArithValue(smem[fx.Int32(OFF_NM) + fx.Int32(2)])
        fx.rocdl.s_barrier()

        bk = fx.arith.mulf(beta_f, k_f)
        qs = fx.arith.mulf(q_f, scale_f)
        decay_v = mlir_vector.broadcast(f32v4, fmath.exp(g_f))
        sb = fx.arith.mulf(scale_f, beta_f)

        # ---------- main phase 1: decay + stage kv/od partials ----------
        s_row = s_head + tid * fx.Int32(VV) + col0
        dec = [None] * NVEC
        for i in range_constexpr(NVEC):
            dec[i] = fx.arith.mulf(
                buffer_load(s_r, s_row + fx.Int32(i * 4), vec_width=4, dtype=f32),
                decay_v)
            kd = fx.arith.mulf(mlir_vector.broadcast(f32v4, k_f), dec[i])
            od = fx.arith.mulf(mlir_vector.broadcast(f32v4, qs), dec[i])
            for j in range_constexpr(4):
                smem[fx.Int32(OFF_KVP) + tid * fx.Int32(BV) + fx.Int32(i * 4 + j)] = _lane(kd, j)
                smem[fx.Int32(OFF_ODP) + tid * fx.Int32(BV) + fx.Int32(i * 4 + j)] = _lane(od, j)
        fx.rocdl.s_barrier()

        # ---------- main phase 2: column reductions ----------
        # threads 0..31: kv[j]; 32..63: od[j]; 64..71: u staging prep (idle ok)
        if tid < fx.Int32(64):
            is_od = tid >= fx.Int32(32)
            j = tid % fx.Int32(32)
            base = fx.Int32(OFF_ODP) if is_od else fx.Int32(OFF_KVP)
            acc = fx.arith.ArithValue(smem[base + fx.Int32(j)])
            for r in range_constexpr(1, KK):
                acc = fx.arith.addf(
                    acc, fx.arith.ArithValue(smem[base + fx.Int32(r * BV) + fx.Int32(j)]))
            if is_od:
                smem[fx.Int32(OFF_ODR) + fx.Int32(j)] = acc
            else:
                smem[fx.Int32(OFF_KVR) + fx.Int32(j)] = acc
        fx.rocdl.s_barrier()

        # ---------- main phase 3: u, o ----------
        if tid < fx.Int32(8):
            for j in range_constexpr(4):
                col = tid * fx.Int32(4) + fx.Int32(j)
                kv_j = fx.arith.ArithValue(smem[fx.Int32(OFF_KVR) + col])
                v_j = fx.arith.extf(
                    f32, buffer_load(v_r, v_base + col0 + col, vec_width=1, dtype=bf16))
                u_j = fx.arith.subf(v_j, kv_j)
                od_j = fx.arith.ArithValue(smem[fx.Int32(OFF_ODR) + col])
                o_j = fx.arith.addf(
                    od_j, fx.arith.mulf(sb, fx.arith.mulf(qk_n, u_j)))
                buffer_store(fx.arith.trunc_f(bf16, o_j), o_r, v_base + col0 + col)
                smem[fx.Int32(OFF_U) + col] = u_j
        fx.rocdl.s_barrier()

        # ---------- main phase 4: state update + store (registers -> global) ----
        for i in range_constexpr(NVEC):
            sv = dec[i]
            out = mlir_vector.broadcast(f32v4, fx.arith.ArithValue(smem[fx.Int32(0)]))
            # build vec4 via lane-wise scalar ops (no smem round trip)
            for j in range_constexpr(4):
                u_j = fx.arith.ArithValue(smem[fx.Int32(OFF_U) + fx.Int32(i * 4 + j)])
                dec_j = _lane(sv, j)
                new_s = fx.arith.addf(dec_j, fx.arith.mulf(bk, u_j))
                out = _repl_lane(out, j, new_s)
            buffer_store(out, so_r, s_row + fx.Int32(i * 4))

    @flyc.jit
    def launch(
        Q: fx.Tensor, KB: fx.Tensor, GB: fx.Tensor, Vvin: fx.Tensor,
        BETA: fx.Tensor, S_IN: fx.Tensor, S_OUT: fx.Tensor, O: fx.Tensor,
        SCALE_T: fx.Tensor, BHV: fx.Int32, stream: fx.Stream,
    ):
        kda_decode_step_kern(
            Q, KB, GB, Vvin, BETA, S_IN, S_OUT, O, SCALE_T
        ).launch(grid=(BHV, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


def _repl_lane(v, j, scalar):
    """Replace lane j of a f32v4 with scalar (vector.insert)."""
    from flydsl._mlir.dialects import vector as mv
    return fx.arith.ArithValue(mv.insert(scalar, v, [], [j]))


def decode_step(q, k, g, v, beta, S, scale=_SCALE):
    B, _, H, _ = q.shape
    launch = _compile_decode_kernel()
    s_out = torch.empty_like(S)
    o = torch.empty((B, 1, H, VV), dtype=torch.bfloat16, device=q.device)
    scale_t = torch.tensor([float(scale)], dtype=torch.float32, device=q.device)
    launch(
        q.reshape(B * H, KK).contiguous(), k.reshape(B * H, KK).contiguous(),
        g.reshape(B * H, KK).contiguous(), v.reshape(B * H, VV).contiguous(),
        beta.reshape(B * H).contiguous(),
        S.contiguous(), s_out, o, scale_t, B * H * NV, torch.cuda.current_stream(),
    )
    return o, s_out


def get_forward():
    def fwd(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
            cu_seqlens=None):
        assert q.shape[1] == 1 and q.shape[-1] == KK, "flydsl-decode-v4: T=1, K=128 only"
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
