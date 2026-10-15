"""KDA decode spec: exact-real reference math in the checker's abstraction.

Same convention as check.py: loads become named Real constants keyed by
(buffer spec-name, linear element index); f_exp / f_rsqrt are the SHARED
uninterpreted functions (abstracting approximate transcendental ops — see
kagent/formal/__init__). The spec therefore proves structure (index maps,
reduction coverage, factors), not precision; precision stays with the numeric
verifier (guidelines 02).

`off` maps spec-facing buffer names to the linear-index base for the queried
head: {"Q": nh*K, "K": nh*K, "G": nh*K, "V": nh*V, "S_IN": nh*K*V, "BETA": nh}.

KDA decode recurrence (per head, token t; K=V=128), mirroring ref_fp64:
    rq2 = sum_i q_i^2; rk2 = sum_i k_i^2
    qh_i = q_i * f_rsqrt(rq2); kh_i = k_i * f_rsqrt(rk2)     (L2 normalization)
    dec_i = f_exp(g_i); u_j = v_j - sum_i kh_i*dec_i*S0[i][j]
    o[j]  = scale * sum_i qh_i*dec_i*S0[i][j] + scale*beta*(sum_i qh_i*kh_i)*u_j
    S1[i][j] = dec_i*S0[i][j] + beta*kh_i*u_j
"""
from __future__ import annotations


def build_fns(z3):
    f_exp = z3.Function("f_exp", z3.RealSort(), z3.RealSort())
    f_rsqrt = z3.Function("f_rsqrt", z3.RealSort(), z3.RealSort())
    return f_exp, f_rsqrt


class KdaSpec:
    name = "kda_decode"

    def __init__(self, K: int = 128, V: int = 128, layout: str = "k_first"):
        self.K, self.V = K, V
        self.layout = layout

    def _state_idx(self, i: int, j: int) -> int:
        return j * self.K + i if self.layout == "v_first" else i * self.V + j

    def expected_o(self, z3, mk, f_exp, f_rsqrt, j: int, off: dict):
        """Returns (z3 expr for o[j], multiset of expected (buffer, idx) loads)."""
        loads: dict[tuple[str, int], int] = {}

        def m(buf, idx):
            key = (buf, idx)
            loads[key] = loads.get(key, 0) + 1
            return mk(buf, idx)

        scale = m("SCALE_T", 0)
        beta = m("BETA", off["BETA"])
        v_j = m("V", off["V"] + j)
        q, k, g, s0 = {}, {}, {}, {}
        for i in range(self.K):
            q[i] = m("Q", off["Q"] + i)
            k[i] = m("K", off["K"] + i)
            g[i] = m("G", off["G"] + i)
            s0[i] = m("S_IN", off["S_IN"] + self._state_idx(i, j))
        rq2 = sum(q[i] * q[i] for i in range(self.K))
        rk2 = sum(k[i] * k[i] for i in range(self.K))
        inv_q = f_rsqrt(rq2)
        inv_k = f_rsqrt(rk2)
        qh = {i: q[i] * inv_q for i in range(self.K)}
        kh = {i: k[i] * inv_k for i in range(self.K)}
        qk = sum(qh[i] * kh[i] for i in range(self.K))
        od = sum(qh[i] * f_exp(g[i]) * s0[i] for i in range(self.K))
        kv = sum(kh[i] * f_exp(g[i]) * s0[i] for i in range(self.K))
        u = v_j - kv
        return scale * od + scale * beta * qk * u, loads

    def expected_state(self, z3, mk, f_exp, f_rsqrt, i: int, j: int, off: dict):
        loads: dict[tuple[str, int], int] = {}

        def m(buf, idx):
            key = (buf, idx)
            loads[key] = loads.get(key, 0) + 1
            return mk(buf, idx)

        # SCALE_T is not in the state spec: the state math does not use scale
        beta = m("BETA", off["BETA"])
        v_j = m("V", off["V"] + j)
        g_i = m("G", off["G"] + i)
        k_i = m("K", off["K"] + i)
        s0 = m("S_IN", off["S_IN"] + self._state_idx(i, j))
        rk2 = 0
        khat_sum = 0
        for mm in range(self.K):
            k_m = m("K", off["K"] + mm)
            g_m = m("G", off["G"] + mm)
            s0_m = m("S_IN", off["S_IN"] + self._state_idx(mm, j))
            rk2 = rk2 + k_m * k_m
            khat_sum = khat_sum + k_m * f_exp(g_m) * s0_m
        inv_k = f_rsqrt(rk2)
        u = v_j - khat_sum * inv_k
        return f_exp(g_i) * s0 + beta * (k_i * inv_k) * u, loads