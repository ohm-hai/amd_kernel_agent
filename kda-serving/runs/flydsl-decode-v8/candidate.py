#!/usr/bin/env python3
"""Candidate: flydsl-decode-v8 — native in-place indexed state pool.

v8 = v7 (PROVED math, 256-thread column-split, zero-barrier) + the serving
interface done natively:
- kernel takes an IDX buffer (int32 [B]); state addressing is
  slot = IDX[b]; s_head = slot*(H*K*V) + h*(K*V) elements
- the pool is passed as BOTH S_IN and S_OUT: each state element is read and
  written only by its owning lane (col-group w owns element (i,c) iff
  c//8 == w), so in-place update has no cross-lane hazard
- decode_pool(q,k,v,g,beta,pool,idx) updates pool[idx] in place, no host sync
  (graph-capturable); no gather/scatter passes
- H is baked per compile and cached per H (H is model-fixed: 96 for Kimi K3;
  the verifier compiles H in {1, 8, 96}); this keeps the slot stride out of
  the runtime index chain entirely
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

INPLACE = True         # native serving interface: pool updated in place

_SCALE_T = {}          # device -> scale tensor, created once (outside capture)


def _scale_t(dev):
    t = _SCALE_T.get(dev)
    if t is None:
        t = torch.tensor([float(_SCALE)], dtype=torch.float32, device=dev)
        _SCALE_T[dev] = t
    return t


def _lane(v, j):
    return fx.arith.ArithValue(mlir_vector.extract(v, [], [j]))


def _tree(val):
    for i in range_constexpr(6):
        off, width = _TREE[i]
        val = fx.arith.addf(val, val.shuffle_xor(fx.Int32(off), fx.Int32(width)))
    return val


def _repl(v, j, scalar):
    from flydsl._mlir.dialects import vector as mv
    return fx.arith.ArithValue(mv.insert(scalar, v, [], [j]))


@functools.lru_cache(maxsize=8)
def _compile_decode_kernel(H):
    """Compile the decode kernel with the slot stride baked for head count H."""
    SLOT_STRIDE = H * KK * VV           # elements per pool slot

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def kda_decode_step_kern(
        Q: fx.Tensor, KB: fx.Tensor, GB: fx.Tensor, Vvin: fx.Tensor,
        BETA: fx.Tensor, S_IN: fx.Tensor, S_OUT: fx.Tensor, O: fx.Tensor,
        IDX: fx.Tensor, SCALE_T: fx.Tensor,
    ):
        i32 = fx.T.i32()
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

        f32 = fx.T.f32()
        bf16 = fx.T.bf16()
        f32v4 = fx.T.VectorType.get([4], f32)

        wf = tid // fx.Int32(64)
        lane = tid % fx.Int32(64)
        col0 = iv * fx.Int32(BV) + wf * fx.Int32(CG)   # this thread's col-group base
        row0 = lane * fx.Int32(2)                       # rows 2l, 2l+1
        head_base = nh * fx.Int32(KK)
        v_base = nh * fx.Int32(VV)

        # ---- in-place indexed pool: slot from IDX, per-head state base ----
        slot = buffer_load(idx_r, bidx, vec_width=1, dtype=i32)
        s_head = slot * fx.Int32(SLOT_STRIDE) + h * fx.Int32(KK * VV)

        q_a = fx.arith.extf(f32, buffer_load(q_r, head_base + row0, vec_width=1, dtype=bf16))
        q_b = fx.arith.extf(f32, buffer_load(q_r, head_base + row0 + fx.Int32(1), vec_width=1, dtype=bf16))
        k_a = fx.arith.extf(f32, buffer_load(k_r, head_base + row0, vec_width=1, dtype=bf16))
        k_b = fx.arith.extf(f32, buffer_load(k_r, head_base + row0 + fx.Int32(1), vec_width=1, dtype=bf16))
        g_a = fx.arith.extf(f32, buffer_load(g_r, head_base + row0, vec_width=1, dtype=bf16))
        g_b = fx.arith.extf(f32, buffer_load(g_r, head_base + row0 + fx.Int32(1), vec_width=1, dtype=bf16))
        beta_f = fx.arith.extf(f32, buffer_load(b_r, nh, vec_width=1, dtype=bf16))
        scale_f = buffer_load(sc_r, fx.Int32(0), vec_width=1, dtype=f32)

        # ---- L2N + qk: wavefront covers all 128 rows -> tree = FULL sum, 0 barriers
        sq = _tree(fx.arith.addf(fx.arith.mulf(q_a, q_a), fx.arith.mulf(q_b, q_b)))
        sk = _tree(fx.arith.addf(fx.arith.mulf(k_a, k_a), fx.arith.mulf(k_b, k_b)))
        qk = _tree(fx.arith.addf(fx.arith.mulf(q_a, k_a), fx.arith.mulf(q_b, k_b)))
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
        sbqk_v = mlir_vector.broadcast(f32v4, sb_qk)

        # ---- state rows: 2 rows x 8 cols = 4 vec4 registers ----
        s_r0 = s_head + row0 * fx.Int32(VV) + col0
        s_r1 = s_head + (row0 + fx.Int32(1)) * fx.Int32(VV) + col0
        d_a = [None] * 2
        d_b = [None] * 2
        for i in range_constexpr(2):
            d_a[i] = fx.arith.mulf(
                buffer_load(s_r, s_r0 + fx.Int32(i * 4), vec_width=4, dtype=f32), dec_a)
            d_b[i] = fx.arith.mulf(
                buffer_load(s_r, s_r1 + fx.Int32(i * 4), vec_width=4, dtype=f32), dec_b)

        # ---- kv/od trees: per column c, partial = k_a*dec_a[c] + k_b*dec_b[c] ----
        kv = [None] * CG
        od = [None] * CG
        for i in range_constexpr(2):
            for j in range_constexpr(4):
                c = i * 4 + j
                kv[c] = _tree(fx.arith.addf(
                    fx.arith.mulf(k_a, _lane(d_a[i], j)),
                    fx.arith.mulf(k_b, _lane(d_b[i], j))))
        for i in range_constexpr(2):
            for j in range_constexpr(4):
                c = i * 4 + j
                od[c] = _tree(fx.arith.addf(
                    fx.arith.mulf(qs_a, _lane(d_a[i], j)),
                    fx.arith.mulf(qs_b, _lane(d_b[i], j))))

        # ---- u = v - kv ; o = od + sb*qk*u ; o stores (lanes 0..7) ----
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

        bf16v4 = fx.T.VectorType.get([4], bf16)
        # ---- o store: lane 0 builds 2 vec4 from its registers, 2 stores ----
        if lane < fx.Int32(1):
            for half in range_constexpr(2):
                ov = mlir_vector.broadcast(f32v4, o[half * 4])
                for j in range_constexpr(4):
                    ov = _repl(ov, j, o[half * 4 + j])
                buffer_store(fx.arith.trunc_f(bf16v4, ov),
                             o_r, v_base + col0 + fx.Int32(half * 4))

        # ---- state update + store: 4 vec4, registers -> global (32B runs) ----
        for half in range_constexpr(2):
            for rr in range_constexpr(2):
                base = d_a[half] if rr == 0 else d_b[half]
                s_row = s_r0 if rr == 0 else s_r1
                bk = bk_a if rr == 0 else bk_b
                out = mlir_vector.broadcast(f32v4, _lane(base, 0))
                for j in range_constexpr(4):
                    c = half * 4 + j
                    out = _repl(out, j,
                                fx.arith.addf(_lane(base, j), fx.arith.mulf(bk, u[c])))
                buffer_store(out, so_r, s_row + fx.Int32(half * 4))

    @flyc.jit
    def launch(
        Q: fx.Tensor, KB: fx.Tensor, GB: fx.Tensor, Vvin: fx.Tensor,
        BETA: fx.Tensor, S_IN: fx.Tensor, S_OUT: fx.Tensor, O: fx.Tensor,
        IDX: fx.Tensor, SCALE_T: fx.Tensor, BHV: fx.Int32, stream: fx.Stream,
    ):
        kda_decode_step_kern(
            Q, KB, GB, Vvin, BETA, S_IN, S_OUT, O, IDX, SCALE_T
        ).launch(grid=(BHV, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


def decode_pool(q, k, v, g, beta, pool, idx):
    """Native serving step: update pool[idx] in place; returns o [B,1,H,V] bf16.

    Graph-capturable: no host sync, no data-dependent shapes; the o allocation
    lands in the CUDA-graph pool on capture. q/k/v/g/beta are contiguous views.
    """
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
    """Verifier path: initial_state [B,H,K,V] updated in place (INPLACE=True)."""
    def fwd(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
            cu_seqlens=None):
        assert q.shape[1] == 1 and q.shape[-1] == KK, "flydsl-decode-v8: T=1, K=128 only"
        assert initial_state is not None, "v8 serving kernel updates a state in place"
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
    q = torch.randn(B, T, H, KK, device=dev).bfloat16()
    k = torch.randn(B, T, H, KK, device=dev).bfloat16()
    v = torch.randn(B, T, H, VV, device=dev).bfloat16()
    g = -torch.rand(B, T, H, KK, device=dev).bfloat16().abs()
    beta = torch.rand(B, T, H, device=dev).bfloat16().sigmoid()
    idx = torch.tensor([3, 1], dtype=torch.int32, device=dev)   # NON-identity slots
    slots = idx.tolist()
    S0 = (torch.randn(P, H, KK, VV, device=dev) * 0.1)[slots].clone()

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