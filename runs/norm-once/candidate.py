#!/usr/bin/env python3
"""Candidate: norm-once — baseline computes ln1(x) 3x per block (q,k,v); compute once.

Parent: baseline-eager. No torch.compile, no aiter JIT — pure eager rewrite.
KDA loop: validate (allclose vs baseline, shared weights) + profile + TraceLens.
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

    def forward(self, x, norm_once):
        if norm_once:
            x1 = self.ln1(x)  # computed ONCE (baseline computes it 3x)
            a, _ = self.attn(x1, x1, x1, need_weights=False)
        else:
            a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, tokens, norm_once=True):
        x = self.embed(tokens)
        for b in self.blocks:
            x = b(x, norm_once)
        return self.head(x)


def validate():
    torch.manual_seed(0)
    base = Model().cuda().eval()
    cand = Model().cuda().eval()
    cand.load_state_dict(base.state_dict())
    torch.manual_seed(42)
    tok = torch.randint(0, V, (4, 256), device="cuda")
    with torch.no_grad():
        lb = base(tok, norm_once=False)
        lc = cand(tok, norm_once=True)
    ok = torch.allclose(lb.float(), lc.float(), atol=1e-2, rtol=1e-3)
    print("norm-once vs baseline allclose:", ok)
    assert ok
    print("VALIDATION PASSED")


def profile(out_path):
    torch.manual_seed(0)
    model = Model().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    from torch.profiler import ProfilerActivity, profile

    tok = torch.randint(0, V, (16, 512), device="cuda")
    for i in range(6):
        if i == 2:
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.start()
        loss = torch.nn.functional.cross_entropy(model(tok).view(-1, V), tok.view(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i == 2:
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