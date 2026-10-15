#!/usr/bin/env python3
"""Candidate: flydsl-decode-v5 — wavefront shuffle-tree FlyDSL KDA decode step.

Profiling-driven redesign (v4: 15.7us kernel vs FLA 6.1us at B=1; 819 vs 310 at B=128):
- ALL cross-row reductions become in-wavefront shuffle_xor trees (verified exact
  in probe): L2N (sq,sk,qk), kv = k_hat.S_decay, od = qs.S_decay
- ZERO smem staging in the hot path; ZERO serial FADD chains; 1 barrier/token
  (v4: 6, v2: 16); o from fused algebra o = od + scale*beta*qk*u
- every lane redundantly computes u/o (v loads L2-cached) -> no u staging

Layout: grid (B*H*NV), NV=4 col-tiles of 32; 128 threads (2 wavefronts x 64);
thread t = tile row t (32 cols, 8 vec4 registers).
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
BV = VV // NV
NVEC = BV // 4
_SCALE = KK ** -0.5
_TREE = [(32, 64), (16, 64), (8, 64), (4, 64), (2, 64), (1, 64)]  # xor offsets

# smem: L2N handoff (6) + kv/od wf-merge (64+64)
SMEM_LEN = 6 + 128


def _lane(v, j):
    return fx.arith.ArithValue(mlir_vector.extract(v, [], [j]))


def _wf_tree_sum(val):
    """Full wavefront xor-tree sum; returns total in EVERY lane of the wavefront."""
    for i in range_constexpr(6):
        off, width = _TREE[i]
        other = val.shuffle_xor(fx.Int32(off), fx.Int32(width))
        val = fx.arith.addf(val, other)
    return val


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
        wf = tid // fx.Int32(64)

        k_f = fx.arith.extf(f32, buffer_load(k_r, row_base + tid, vec_width=1, dtype=bf16))
        g_f = fx.arith.extf(f32, buffer_load(g_r, row_base + tid, vec_width=1, dtype=bf16))
        q_f = fx.arith.extf(f32, buffer_load(q_r, row_base + tid, vec_width=1, dtype=bf16))
        beta_f = fx.arith.extf(f32, buffer_load(b_r, nh, vec_width=1, dtype=bf16))
        scale_f = buffer_load(sc_r, fx.Int32(0), vec_width=1, dtype=f32)

        # ---- L2N + qk via wavefront trees ----
        sq = _wf_tree_sum(fx.arith.mulf(q_f, q_f))
        sk = _wf_tree_sum(fx.arith.mulf(k_f, k_f))
        qk = _wf_tree_sum(fx.arith.mulf(q_f, k_f))
        if tid % fx.Int32(64) == fx.Int32(0):
            smem[fx.Int32(wf * 3)] = sq
            smem[fx.Int32(wf * 3 + 1)] = sk
            smem[fx.Int32(wf * 3 + 2)] = qk
        fx.rocdl.s_barrier()
        sq = fx.arith.addf(
            fx.arith.ArithValue(smem[fx.Int32(0)]), fx.arith.ArithValue(smem[fx.Int32(3)]))
        sk = fx.arith.addf(
            fx.arith.ArithValue(smem[fx.Int32(1)]), fx.arith.ArithValue(smem[fx.Int32(4)]))
        qk_raw = fx.arith.addf(
            fx.arith.ArithValue(smem[fx.Int32(2)]), fx.arith.ArithValue(smem[fx.Int32(5)]))
        inv_q = fmath.rsqrt(sq)
        inv_k = fmath.rsqrt(sk)
        qk_n = fx.arith.mulf(fx.arith.mulf(qk_raw, inv_q), inv_k)
        q_f = fx.arith.mulf(q_f, inv_q)
        k_f = fx.arith.mulf(k_f, inv_k)

        bk = fx.arith.mulf(beta_f, k_f)
        qs = fx.arith.mulf(q_f, scale_f)
        sb_qk = fx.arith.mulf(fx.arith.mulf(scale_f, beta_f), qk_n)
        decay_v = mlir_vector.broadcast(f32v4, fmath.exp(g_f))
        bk_v = mlir_vector.broadcast(f32v4, bk)
        qs_v = mlir_vector.broadcast(f32v4, qs)
        sbqk_v = mlir_vector.broadcast(f32v4, sb_qk)

        # ---- load state row, decay (registers) ----
        s_row = s_head + tid * fx.Int32(VV) + col0
        dec = [None] * NVEC
        for i in range_constexpr(NVEC):
            dec[i] = fx.arith.mulf(
                buffer_load(s_r, s_row + fx.Int32(i * 4), vec_width=4, dtype=f32),
                decay_v)

        # ---- kv tree (per wavefront) + cross-wf merge via smem ----
        OFF_KVM = 6
        OFF_ODM = 6 + 64
        for i in range_constexpr(NVEC):
            for j in range_constexpr(4):
                part = fx.arith.mulf(k_f, _lane(dec[i], j))
                tot = _wf_tree_sum(part)
                if tid % fx.Int32(64) == fx.Int32(0):
                    smem[fx.Int32(OFF_KVM) + fx.Int32(wf * BV) + fx.Int32(i * 4 + j)] = tot
        for i in range_constexpr(NVEC):
            for j in range_constexpr(4):
                part = fx.arith.mulf(qs, _lane(dec[i], j))
                tot = _wf_tree_sum(part)
                if tid % fx.Int32(64) == fx.Int32(0):
                    smem[fx.Int32(OFF_ODM) + fx.Int32(wf * BV) + fx.Int32(i * 4 + j)] = tot
        fx.rocdl.s_barrier()
        kv = [None] * BV
        od = [None] * BV
        for i in range_constexpr(NVEC):
            for j in range_constexpr(4):
                c = fx.Int32(i * 4 + j)
                kv[i * 4 + j] = fx.arith.addf(
                    fx.arith.ArithValue(smem[fx.Int32(OFF_KVM) + c]),
                    fx.arith.ArithValue(smem[fx.Int32(OFF_KVM) + fx.Int32(BV) + c]))
                od[i * 4 + j] = fx.arith.addf(
                    fx.arith.ArithValue(smem[fx.Int32(OFF_ODM) + c]),
                    fx.arith.ArithValue(smem[fx.Int32(OFF_ODM) + fx.Int32(BV) + c]))

        # ---- u = v - kv ; o = od + sb*qk*u (every lane redundantly) ----
        u = [None] * BV
        for i in range_constexpr(NVEC):
            vv_bf = buffer_load(v_r, v_base + col0 + fx.Int32(i * 4), vec_width=4, dtype=bf16)
            vv = fx.arith.extf(f32v4, vv_bf)
            for j in range_constexpr(4):
                v_j = _lane(vv, j)
                u_j = fx.arith.subf(v_j, kv[i * 4 + j])
                u[i * 4 + j] = u_j
                o_j = fx.arith.addf(od[i * 4 + j], fx.arith.mulf(sb_qk, u_j))
                if tid < fx.Int32(BV):
                    buffer_store(fx.arith.trunc_f(bf16, o_j),
                                 o_r, v_base + col0 + fx.Int32(i * 4 + j))

        # ---- state update + store (registers -> global, contiguous row) ----
        for i in range_constexpr(NVEC):
            out = mlir_vector.broadcast(f32v4, _lane(dec[i], 0))
            for j in range_constexpr(4):
                out = _repl_lane(out, j,
                                 fx.arith.addf(_lane(dec[i], j), fx.arith.mulf(bk, u[i * 4 + j])))
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
        assert q.shape[1] == 1 and q.shape[-1] == KK, "flydsl-decode-v5: T=1, K=128 only"
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
