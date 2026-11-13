"""tiny-prefetch: the serving side of a sparse MoE.

Two-layer MoE on a synthetic 4-task problem, plus a lookahead probe that
predicts which experts layer 1 will want, one layer ahead of time. serve.py
uses the probe to prefetch experts while layer 0 is still computing.

Nothing here is big. The point is the mechanism, not the scale.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_TASKS = 4
N_FEATURES = 8
INPUT_DIM = N_TASKS + N_FEATURES  # task one-hot + continuous features
HIDDEN = 32
N_EXPERTS = 8
TOP_K = 2
N_LAYERS = 2
SEED = 0


def make_data(n, seed=SEED):
    """Synthetic 4-task binary problem. Each task has its own labeling rule,
    so the router has something real to specialize on."""
    rng = np.random.default_rng(seed)
    task = rng.integers(0, N_TASKS, size=n)
    x = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    y[task == 0] = (x[task == 0, 0] + x[task == 0, 1] > 0)
    y[task == 1] = (x[task == 1, 2] * x[task == 1, 3] > 0)
    y[task == 2] = (np.abs(x[task == 2, 4]) > 0.7)
    y[task == 3] = (x[task == 3, 5] - x[task == 3, 6] + 0.5 * x[task == 3, 7] > 0)
    onehot = np.zeros((n, N_TASKS), dtype=np.float32)
    onehot[np.arange(n), task] = 1.0
    inputs = np.concatenate([onehot, x], axis=1)
    return torch.from_numpy(inputs), torch.from_numpy(y), task


class Expert(nn.Module):
    def __init__(self, d=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(self, d=HIDDEN, n_experts=N_EXPERTS, top_k=TOP_K):
        super().__init__()
        self.router = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([Expert(d) for _ in range(n_experts)])
        self.top_k = top_k
        self.n_experts = n_experts

    def forward(self, x):
        # x: [B, d]. Returns output, router probs, and the chosen expert ids.
        logits = self.router(x)
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            # tokens that routed to expert e (in either slot)
            mask = (topi == e).any(dim=-1)
            if not mask.any():
                continue
            w = torch.where(topi[mask] == e, topv[mask], torch.zeros_like(topv[mask])).sum(dim=-1)
            out[mask] = out[mask] + w.unsqueeze(-1) * self.experts[e](x[mask])
        return out, probs, topi


class TinyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(INPUT_DIM, HIDDEN)
        self.layers = nn.ModuleList([MoELayer() for _ in range(N_LAYERS)])
        self.head = nn.Linear(HIDDEN, 2)

    def forward(self, x):
        h = F.gelu(self.in_proj(x))
        aux = 0.0
        for layer in self.layers:
            out, probs, _ = layer(h)
            h = h + out  # residual, pre-norm would be nicer, this is fine at this size
            # Switch-style load-balancing loss, kept tiny on purpose
            f = (probs.argmax(dim=-1).unsqueeze(-1) == torch.arange(N_EXPERTS)).float().mean(dim=0)
            aux = aux + (f * probs.mean(dim=0)).sum()
        return self.head(h), aux * N_EXPERTS

    def forward_trace(self, x):
        """Same forward, but also returns the per-layer expert choices and
        the hidden state entering layer 0 (the probe's input)."""
        h_in = F.gelu(self.in_proj(x))
        h = h_in
        choices = []
        for layer in self.layers:
            out, _, topi = layer(h)
            choices.append(topi)
            h = h + out
        return self.head(h), h_in, choices


class LookaheadProbe(nn.Module):
    """One linear layer: hidden state entering layer 0 -> which experts
    layer 1 will pick. If routing is mostly task-determined, this is easy;
    that is the point — predictability is what makes prefetching possible."""

    def __init__(self):
        super().__init__()
        self.probe = nn.Linear(HIDDEN, N_EXPERTS)

    def forward(self, h):
        return self.probe(h)
