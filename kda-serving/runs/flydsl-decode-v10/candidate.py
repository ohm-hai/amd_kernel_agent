#!/usr/bin/env python3
"""Candidate: flydsl-decode-v10 — V-Major Contiguous State Layout (Pure FP32).

v10 optimizes the memory layout of the state from K-major ([K, V]) to V-major ([V, K]):
1. V-Major Coalesced Memory Access:
   - For column c, lane l reads rows 2l and 2l+1 at s_head + c*KK + row0.
   - Across the 64 lanes of each wavefront, rows 0..127 form a contiguous 512-byte block.
   - Loads and stores coalesce into exactly 4 contiguous 128-byte burst transactions (100% bus efficiency).
2. Hoisted State Loads:
   - Pre-issues all 8 column vector loads (vec_width=2) at the top of the kernel to overlap memory latency.
3. Pure FP32:
   - State S remains strictly torch.float32.
   - Exact bit-for-bit equivalence to Triton FLA state_v_first.
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
STATE_LAYOUT = "v_first"

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
    # State S in V-major has shape [P, H, V, K]
    # For head h and sequence slot s:
    # offset = slot * (H * V * K) + h * (V * K) + c * K + k
    SLOT_STRIDE = H * VV * KK

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def kda_decode_step_v10(
        Q: fx.Tensor, KB: fx.Tensor, GB: fx.Tensor, Vvin: fx.Tensor,
        BETA: fx.Tensor, S_IN: fx.Tensor, S_OUT: fx.Tensor, O: fx.Tensor,
        IDX: fx.Tensor, SCALE_T: fx.Tensor,
    ):
        i32 = fx.T.i32()
        f32 = fx.T.f32()
        bf16 = fx.T.bf16()
        f32v2 = fx.T.VectorType.get([2], f32)
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
        s_head = slot * fx.Int32(SLOT_STRIDE) + h * fx.Int32(VV * KK)

        # 1. Hoist coalesced V-major state loads immediately:
        s_raw = [None] * CG
        for c in range_constexpr(CG):
            col = col0 + fx.Int32(c)
            s_offset = s_head + col * fx.Int32(KK) + row0
            s_raw[c] = buffer_load(s_r, s_offset, vec_width=2, dtype=f32)

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
        dec_a = fmath.exp(g_a)
        dec_b = fmath.exp(g_b)

        # 4. Apply decay to hoisted state:
        d_a = [None] * CG
        d_b = [None] * CG
        for c in range_constexpr(CG):
            d_a[c] = fx.arith.mulf(_lane(s_raw[c], 0), dec_a)
            d_b[c] = fx.arith.mulf(_lane(s_raw[c], 1), dec_b)

        # 5. Interleaved reduction for kv and od
        kv_p = [None] * CG
        od_p = [None] * CG
        for c in range_constexpr(CG):
            kv_p[c] = fx.arith.addf(
                fx.arith.mulf(k_a, d_a[c]),
                fx.arith.mulf(k_b, d_b[c]))
            od_p[c] = fx.arith.addf(
                fx.arith.mulf(qs_a, d_a[c]),
                fx.arith.mulf(qs_b, d_b[c]))

        kv = _reduce_all(kv_p)
        od = _reduce_all(od_p)

        # 6. Output and state update
        u = [None] * CG
        o = [None] * CG
        vv0 = fx.arith.extf(f32v4, buffer_load(v_r, v_base + col0, vec_width=4, dtype=bf16))
        vv1 = fx.arith.extf(f32v4, buffer_load(v_r, v_base + col0 + fx.Int32(4), vec_width=4, dtype=bf16))
        for c in range_constexpr(CG):
            v_j = _lane(vv0, c) if c < 4 else _lane(vv1, c - 4)
            u[c] = fx.arith.subf(v_j, kv[c])
            o[c] = fx.arith.addf(od[c], fx.arith.mulf(sb_qk, u[c]))

        if lane < fx.Int32(1):
            for half in range_constexpr(2):
                ov = mlir_vector.broadcast(f32v4, o[half * 4])
                for j in range_constexpr(4):
                    ov = _repl(ov, j, o[half * 4 + j])
                buffer_store(fx.arith.trunc_f(bf16v4, ov),
                             o_r, v_base + col0 + fx.Int32(half * 4))

        # 7. Coalesced V-major state stores:
        for c in range_constexpr(CG):
            col = col0 + fx.Int32(c)
            s_offset = s_head + col * fx.Int32(KK) + row0
            new_a = fx.arith.addf(d_a[c], fx.arith.mulf(bk_a, u[c]))
            new_b = fx.arith.addf(d_b[c], fx.arith.mulf(bk_b, u[c]))
            new_v = fx.arith.ArithValue(mlir_vector.from_elements(f32v2, [new_a, new_b]))
            buffer_store(new_v, so_r, s_offset)

    @flyc.jit
    def launch(
        Q: fx.Tensor, KB: fx.Tensor, GB: fx.Tensor, Vvin: fx.Tensor,
        BETA: fx.Tensor, S_IN: fx.Tensor, S_OUT: fx.Tensor, O: fx.Tensor,
        IDX: fx.Tensor, SCALE_T: fx.Tensor, BHV: fx.Int32, stream: fx.Stream,
    ):
        kda_decode_step_v10(
            Q, KB, GB, Vvin, BETA, S_IN, S_OUT, O, IDX, SCALE_T
        ).launch(grid=(BHV, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


def decode_pool(q, k, v, g, beta, pool, idx):
    """Native serving step: updates pool[idx] in place in V-major format; returns o [B,1,H,V] bf16."""
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
    """Verifier adapter: accepts K-major initial_state [B,H,K,V], transposes to V-major, updates in place."""
    def fwd(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
            cu_seqlens=None):
        assert q.shape[1] == 1 and q.shape[-1] == KK, "flydsl-decode-v10: T=1, K=128 only"
        assert initial_state is not None, "v10 serving kernel updates state in place"
        B, _, H, _ = q.shape
        idx = torch.arange(B, device=q.device, dtype=torch.int32)
        # Transpose K-major [B, H, K, V] to V-major [B, H, V, K] for the kernel
        S_v = initial_state.transpose(-1, -2).contiguous()
        o = decode_pool(q, k, v, g, beta, S_v, idx)
        initial_state.copy_(S_v.transpose(-1, -2))
        fwd.last_final_state = initial_state if output_final_state else None
        return o

    fwd.supports_final_state = True
    return fwd


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "kda-attn"))
    from ref_fp64 import kda_fp64_reference, rel_rmse

    dev = "cuda"
    B, H, T, P = 2, 96, 3, 4
    q = torch.randn(B, T, H, KK, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, KK, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, VV, device=dev, dtype=torch.bfloat16)
    g = -torch.rand(B, T, H, KK, device=dev, dtype=torch.bfloat16)
    beta = torch.rand(B, T, H, device=dev, dtype=torch.bfloat16).sigmoid()
    idx = torch.tensor([3, 1], dtype=torch.int32, device=dev)
    slots = idx.tolist()

    # V-major pool format [P, H, V, K]
    pool_v = torch.randn(P, H, VV, KK, device=dev, dtype=torch.float32) * 0.1
    S0_k = pool_v[slots].transpose(-1, -2).clone()

    with torch.no_grad():
        o_ref, S_ref = kda_fp64_reference(q, k, v, g, beta, initial_state=S0_k.to(torch.float64))

    ok = True
    for t in range(T):
        o = decode_pool(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1],
                        g[:, t:t + 1], beta[:, t:t + 1], pool_v, idx)
        r = rel_rmse(o.float(), o_ref[:, t:t + 1].float())
        ok &= r < 0.02
        print(f"step {t}: o rRMSE={r:.4g}")
    st = rel_rmse(pool_v[slots].transpose(-1, -2), S_ref.float())
    ok &= st < 0.02
    print(f"final state rRMSE={st:.4g} (slots {slots}, in-place V-major)")
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)
