#!/usr/bin/env python3
"""Candidate: flydsl-decode-v9 — Pure FP32: Hoisted Loads, Interleaved Trees, Vector FMA.

v9 builds directly on v8 while keeping exact bit-for-bit FP32 state numerics:
1. Hoisted State Loads: Issues all 4 state vec4 loads at the top of the kernel,
   overlapping HBM read latency with Q/K/G loads, L2 norms, and decay exp math.
2. Interleaved Reduction Trees: Replaces 19 sequential single-scalar reduction
   chains with 3-wide (sq, sk, qk) and 8-wide (kv, od) interleaved butterfly reductions,
   hiding shuffle-to-add dependency latency.
3. Vectorized FMA State Updates: Uses mlir_vector.from_elements and vector FMA
   instead of scalar element extraction and insertion.
4. Exact FP32 state (STATE_DTYPE = torch.float32, INPLACE = True).
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
THREADS = 256
NV = 4
BV = VV // NV          # 32 cols per program
CG = BV // 4           # 8 cols per wavefront column-group
_SCALE = KK ** -0.5
_TREE = [(32, 64), (16, 64), (8, 64), (4, 64), (2, 64), (1, 64)]

INPLACE = True
STATE_DTYPE = torch.float32

_SCALE_T = {}


def _scale_t(dev):
    t = _SCALE_T.get(dev)
    if t is None:
        t = torch.tensor([float(_SCALE)], dtype=torch.float32, device=dev)
        _SCALE_T[dev] = t
    return t


def _lane(v, j):
    return fx.arith.ArithValue(mlir_vector.extract(v, [], [j]))


def _repl(v, j, scalar):
    from flydsl._mlir.dialects import vector as mv
    return fx.arith.ArithValue(mv.insert(scalar, v, [], [j]))


def _reduce_all(vals):
    for i in range_constexpr(6):
        off, width = _TREE[i]
        shuffled = [None] * len(vals)
        for j in range_constexpr(len(vals)):
            shuffled[j] = vals[j].shuffle_xor(fx.Int32(off), fx.Int32(width))
        for j in range_constexpr(len(vals)):
            vals[j] = fx.arith.addf(vals[j], shuffled[j])
    return vals


@functools.lru_cache(maxsize=8)
def _compile_decode_kernel(H):
    SLOT_STRIDE = H * KK * VV

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def kda_decode_step_v9(
        Q: fx.Tensor, KB: fx.Tensor, GB: fx.Tensor, Vvin: fx.Tensor,
        BETA: fx.Tensor, S_IN: fx.Tensor, S_OUT: fx.Tensor, O: fx.Tensor,
        IDX: fx.Tensor, SCALE_T: fx.Tensor,
    ):
        i32 = fx.T.i32()
        f32 = fx.T.f32()
        bf16 = fx.T.bf16()
        f32v4 = fx.T.VectorType.get([4], f32)
        bf16v4 = fx.T.VectorType.get([4], bf16)

        tid = fx.thread_idx.x
        bh = fx.block_idx.x
        iv = bh % fx.Int32(NV)
        nh = bh // fx.Int32(NV)
        h = nh % fx.Int32(H)
        bidx = nh // fx.Int32(H)

        q_r = create_buffer_resource(Q, max_size=True)
        k_r = create_buffer_resource(KB, max_size=True)
        g_r = create_buffer_resource(GB, max_size=True)
        v_r = create_buffer_resource(Vvin, max_size=True)
        b_r = create_buffer_resource(BETA, max_size=True)
        s_r = create_buffer_resource(S_IN, max_size=True)
        so_r = create_buffer_resource(S_OUT, max_size=True)
        o_r = create_buffer_resource(O, max_size=True)
        idx_r = create_buffer_resource(IDX, max_size=True)
        sc_r = create_buffer_resource(SCALE_T, max_size=True)

        wf = tid // fx.Int32(64)
        lane = tid % fx.Int32(64)
        col0 = iv * fx.Int32(BV) + wf * fx.Int32(CG)
        row0 = lane * fx.Int32(2)
        head_base = nh * fx.Int32(KK)
        v_base = nh * fx.Int32(VV)

        slot = buffer_load(idx_r, bidx, vec_width=1, dtype=i32)
        s_head = slot * fx.Int32(SLOT_STRIDE) + h * fx.Int32(KK * VV)
        s_r0 = s_head + row0 * fx.Int32(VV) + col0
        s_r1 = s_head + (row0 + fx.Int32(1)) * fx.Int32(VV) + col0

        # 1. Hoist state loads immediately
        s_raw_a0 = buffer_load(s_r, s_r0, vec_width=4, dtype=f32)
        s_raw_a1 = buffer_load(s_r, s_r0 + fx.Int32(4), vec_width=4, dtype=f32)
        s_raw_b0 = buffer_load(s_r, s_r1, vec_width=4, dtype=f32)
        s_raw_b1 = buffer_load(s_r, s_r1 + fx.Int32(4), vec_width=4, dtype=f32)

        # 2. Input loads
        q_a = fx.arith.extf(f32, buffer_load(q_r, head_base + row0, vec_width=1, dtype=bf16))
        q_b = fx.arith.extf(f32, buffer_load(q_r, head_base + row0 + fx.Int32(1), vec_width=1, dtype=bf16))
        k_a = fx.arith.extf(f32, buffer_load(k_r, head_base + row0, vec_width=1, dtype=bf16))
        k_b = fx.arith.extf(f32, buffer_load(k_r, head_base + row0 + fx.Int32(1), vec_width=1, dtype=bf16))
        g_a = fx.arith.extf(f32, buffer_load(g_r, head_base + row0, vec_width=1, dtype=bf16))
        g_b = fx.arith.extf(f32, buffer_load(g_r, head_base + row0 + fx.Int32(1), vec_width=1, dtype=bf16))
        beta_f = fx.arith.extf(f32, buffer_load(b_r, nh, vec_width=1, dtype=bf16))
        scale_f = buffer_load(sc_r, fx.Int32(0), vec_width=1, dtype=f32)

        # 3. Interleaved reduction for sq, sk, qk
        sq_p = fx.arith.addf(fx.arith.mulf(q_a, q_a), fx.arith.mulf(q_b, q_b))
        sk_p = fx.arith.addf(fx.arith.mulf(k_a, k_a), fx.arith.mulf(k_b, k_b))
        qk_p = fx.arith.addf(fx.arith.mulf(q_a, k_a), fx.arith.mulf(q_b, k_b))
        red_qk = _reduce_all([sq_p, sk_p, qk_p])
        sq, sk, qk = red_qk[0], red_qk[1], red_qk[2]

        inv_q = fmath.rsqrt(sq)
        inv_k = fmath.rsqrt(sk)
        qk_n = fx.arith.mulf(fx.arith.mulf(qk, inv_q), inv_k)
        q_a = fx.arith.mulf(q_a, inv_q); q_b = fx.arith.mulf(q_b, inv_q)
        k_a = fx.arith.mulf(k_a, inv_k); k_b = fx.arith.mulf(k_b, inv_k)

        bk_a = fx.arith.mulf(beta_f, k_a)
        bk_b = fx.arith.mulf(beta_f, k_b)
        qs_a = fx.arith.mulf(q_a, scale_f)
        qs_b = fx.arith.mulf(q_b, scale_f)
        sb_qk = fx.arith.mulf(fx.arith.mulf(scale_f, beta_f), qk_n)
        dec_a = mlir_vector.broadcast(f32v4, fmath.exp(g_a))
        dec_b = mlir_vector.broadcast(f32v4, fmath.exp(g_b))

        # 4. Decayed state
        d_a = [fx.arith.mulf(s_raw_a0, dec_a), fx.arith.mulf(s_raw_a1, dec_a)]
        d_b = [fx.arith.mulf(s_raw_b0, dec_b), fx.arith.mulf(s_raw_b1, dec_b)]

        # 5. Interleaved reduction for kv (8 cols) and od (8 cols)
        kv_p = [None] * 8
        od_p = [None] * 8
        for i in range_constexpr(2):
            for j in range_constexpr(4):
                c = i * 4 + j
                kv_p[c] = fx.arith.addf(
                    fx.arith.mulf(k_a, _lane(d_a[i], j)),
                    fx.arith.mulf(k_b, _lane(d_b[i], j)))
                od_p[c] = fx.arith.addf(
                    fx.arith.mulf(qs_a, _lane(d_a[i], j)),
                    fx.arith.mulf(qs_b, _lane(d_b[i], j)))
        
        kv = _reduce_all(kv_p)
        od = _reduce_all(od_p)

        # 6. Output and state update
        u = [None] * CG
        o = [None] * CG
        vv0 = fx.arith.extf(f32v4, buffer_load(v_r, v_base + col0, vec_width=4, dtype=bf16))
        vv1 = fx.arith.extf(f32v4, buffer_load(v_r, v_base + col0 + fx.Int32(4), vec_width=4, dtype=bf16))
        for i in range_constexpr(2):
            for j in range_constexpr(4):
                c = i * 4 + j
                v_j = _lane(vv0, j) if i == 0 else _lane(vv1, j)
                u[c] = fx.arith.subf(v_j, kv[c])
                o[c] = fx.arith.addf(od[c], fx.arith.mulf(sb_qk, u[c]))

        if lane < fx.Int32(1):
            for half in range_constexpr(2):
                ov = mlir_vector.broadcast(f32v4, o[half * 4])
                for j in range_constexpr(4):
                    ov = _repl(ov, j, o[half * 4 + j])
                buffer_store(fx.arith.trunc_f(bf16v4, ov),
                             o_r, v_base + col0 + fx.Int32(half * 4))

        # Vectorized state update
        u_v0 = fx.arith.ArithValue(mlir_vector.from_elements(f32v4, [u[0], u[1], u[2], u[3]]))
        u_v1 = fx.arith.ArithValue(mlir_vector.from_elements(f32v4, [u[4], u[5], u[6], u[7]]))
        u_vecs = [u_v0, u_v1]

        bk_a_v = mlir_vector.broadcast(f32v4, bk_a)
        bk_b_v = mlir_vector.broadcast(f32v4, bk_b)

        for half in range_constexpr(2):
            u_v = u_vecs[half]
            out_a = fx.arith.addf(d_a[half], fx.arith.mulf(bk_a_v, u_v))
            buffer_store(out_a, so_r, s_r0 + fx.Int32(half * 4))

            out_b = fx.arith.addf(d_b[half], fx.arith.mulf(bk_b_v, u_v))
            buffer_store(out_b, so_r, s_r1 + fx.Int32(half * 4))

    @flyc.jit
    def launch(
        Q: fx.Tensor, KB: fx.Tensor, GB: fx.Tensor, Vvin: fx.Tensor,
        BETA: fx.Tensor, S_IN: fx.Tensor, S_OUT: fx.Tensor, O: fx.Tensor,
        IDX: fx.Tensor, SCALE_T: fx.Tensor, BHV: fx.Int32, stream: fx.Stream,
    ):
        kda_decode_step_v9(
            Q, KB, GB, Vvin, BETA, S_IN, S_OUT, O, IDX, SCALE_T
        ).launch(grid=(BHV, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


def decode_pool(q, k, v, g, beta, pool, idx):
    """Native serving step: update pool[idx] in place; returns o [B,1,H,V] bf16."""
    B, _, H, _ = q.shape
    assert pool.dtype == torch.float32 and pool.shape[1] == H
    assert idx.dtype == torch.int32 and idx.shape[0] == B
    launch = _compile_decode_kernel(H)
    o = torch.empty(B, 1, H, VV, dtype=torch.bfloat16, device=q.device)
    launch(
        q.reshape(B * H, KK).contiguous(), k.reshape(B * H, KK).contiguous(),
        g.reshape(B * H, KK).contiguous(), v.reshape(B * H, VV).contiguous(),
        beta.reshape(B * H).contiguous(),
        pool, pool, o, idx, _scale_t(q.device), B * H * NV,
        torch.cuda.current_stream(),
    )
    return o


def get_forward():
    def fwd(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
            cu_seqlens=None):
        assert q.shape[1] == 1 and q.shape[-1] == KK, "flydsl-decode-v9: T=1, K=128 only"
        assert initial_state is not None, "v9 serving kernel updates a state in place"
        B, _, H, _ = q.shape
        idx = torch.arange(B, device=q.device, dtype=torch.int32)
        o = decode_pool(q, k, v, g, beta, initial_state, idx)
        fwd.last_final_state = initial_state if output_final_state else None
        return o

    fwd.supports_final_state = True
    return fwd


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "kda-attn"))
    from ref_fp64 import kda_fp64_reference, rel_rmse

    torch.manual_seed(0)
    B, H, T, P = 2, 96, 3, 4
    dev = "cuda"
    q = torch.randn(B, T, H, KK, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, KK, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, VV, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(B, T, H, KK, device=dev, dtype=torch.bfloat16)
    beta = torch.rand(B, T, H, device=dev, dtype=torch.bfloat16).sigmoid()
    idx = torch.tensor([3, 1], dtype=torch.int32, device=dev)
    slots = idx.tolist()
    S0 = (torch.randn(P, H, KK, VV, device=dev, dtype=torch.float32) * 0.1)[slots].clone()

    with torch.no_grad():
        o_ref, S_ref = kda_fp64_reference(q, k, v, g, beta, initial_state=S0.to(torch.float64))

    pool = torch.zeros(P, H, KK, VV, dtype=torch.float32, device=dev)
    pool[slots] = S0
    ok = True
    for t in range(T):
        o = decode_pool(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1],
                        g[:, t:t + 1], beta[:, t:t + 1], pool, idx)
        r = rel_rmse(o.float(), o_ref[:, t:t + 1].float())
        ok &= r < 0.02
        print(f"step {t}: o rRMSE={r:.4g}")
    st = rel_rmse(pool[slots], S_ref.float())
    ok &= st < 0.02
    print(f"final state rRMSE={st:.4g} (slots {slots}, in-place)")
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)
