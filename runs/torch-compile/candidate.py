#!/usr/bin/env python3
"""Candidate: torch-compile — same eager architecture, whole model compiled with Inductor.

Revises `pt-flash-attn` (rejected: aiter JIT build broken on this box —
aiter 0.1.9 expects ck_tile enums missing from the installed composable_kernel).

KDA loop: validate (allclose vs eager, shared weights) + profile (torch trace).
"""
import sys

import torch
import torch.nn as nn

D, V, L, H = 1024, 8192, 8, 8


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, H, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def validate():
    torch.manual_seed(0)
    eager = Model().cuda().eval()
    comp = Model().cuda().eval()
    comp.load_state_dict(eager.state_dict())
    comp = torch.compile(comp)
    torch.manual_seed(42)
    tok = torch.randint(0, V, (4, 256), device="cuda")
    with torch.no_grad():
        le = eager(tok)
        lc = comp(tok)
    ok = torch.allclose(le.float(), lc.float(), atol=1e-1, rtol=1e-2)
    print("compiled vs eager allclose:", ok)
    assert ok
    print("VALIDATION PASSED")


def profile(out_path):
    torch.manual_seed(0)
    model = Model().cuda()
    cmodel = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    from torch.profiler import ProfilerActivity, profile

    tok = torch.randint(0, V, (16, 512), device="cuda")
    for i in range(7):
        if i == 3:  # compile warmup done, profile steady state
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.start()
        loss = torch.nn.functional.cross_entropy(cmodel(tok).view(-1, V), tok.view(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i == 3:
            prof.stop()
            prof.export_chrome_trace(out_path)
            print("trace:", out_path)
    print("done, loss:", loss.item())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if mode == "validate":
        validate()
    else:
        profile(sys.argv[2] if len(sys.argv) > 2 else "trace.json")