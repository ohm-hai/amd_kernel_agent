#!/usr/bin/env python3
"""Candidate: bug_03_no_l2n (planted bug: v1 incident: missing L2 normalization of q/k; v7-derived — 256-thread column-split, ZERO-barrier decode step.

v7 = v6 + masked o-stores (v6 had 8 lanes each storing all 8 cols = 8x redundant
conflicting stores; v7 predicates each store on lane == column). Layout unchanged:
- 256 threads = 4 wavefronts; wavefront w owns column-group w (8 of 32 cols)
- lane l of each wavefront owns state rows 2l, 2l+1 (16 elems: 4 vec4)
- rows 0..127 all present per wavefront => kv/od/L2N trees complete WITHIN a
  wavefront: zero smem, zero barriers in the whole token step
- ~200 ops/lane, ~40 regs (vs v5 ~900 ops, ~110 regs)
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


@functools.lru_cache(maxsize=4)
def _compile_decode_kernel():
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

        f32 = fx.T.f32()
        bf16 = fx.T.bf16()
        f32v4 = fx.T.VectorType.get([4], f32)

        wf = tid // fx.Int32(64)
        lane = tid % fx.Int32(64)
        col0 = iv * fx.Int32(BV) + wf * fx.Int32(CG)   # this thread's col-group base
        row0 = lane * fx.Int32(2)                       # rows 2l, 2l+1
        head_base = nh * fx.Int32(KK)
        s_head = nh * fx.Int32(KK * VV)
        v_base = nh * fx.Int32(VV)

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
        q_a = q_a; q_b = q_b
        k_a = k_a; k_b = k_b

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
        SCALE_T: fx.Tensor, BHV: fx.Int32, stream: fx.Stream,
    ):
        kda_decode_step_kern(
            Q, KB, GB, Vvin, BETA, S_IN, S_OUT, O, SCALE_T
        ).launch(grid=(BHV, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch


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
        assert q.shape[1] == 1 and q.shape[-1] == KK, "flydsl-decode-v7: T=1, K=128 only"
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
