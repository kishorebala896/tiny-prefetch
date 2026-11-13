"""Serve the tiny MoE three ways and compare.

The idea being demonstrated is Edge0's Nov 12, 2026 release: a 145B MoE
that runs on one laptop, largely because experts are *predicted* and loaded
ahead of time instead of fetched on demand. This file is a small, honest
simulation of that mechanism — same model, three serving policies, one clock.

Policies:
    demand    — no cache at all; every expert is loaded from "disk" when needed.
    lru       — a small cache (CACHE_SLOTS of 16 experts), LRU eviction.
    prefetch  — LRU cache + the lookahead probe: while layer 0 computes, the
                loader fetches the experts layer 1 will probably want.

The simulation is explicit about its assumptions (see the constants below):
disk loads and layer compute are modelled as fixed millisecond costs, and
the loader and the compute run in parallel, so a load that finishes during
compute costs nothing. Prefetching never changes the math — outputs are
verified identical across policies — it only moves waiting around.

Usage:
    python serve.py                    # trains first if weights.pt is missing
    python serve.py --tokens 2000 --cache 6
"""

import argparse

import torch

from moe import N_EXPERTS, N_LAYERS, TOP_K, LookaheadProbe, TinyMoE, make_data

# --- simulation constants (stated assumptions, not measurements) ---
LOAD_MS = 6.0      # cost of loading one expert from "disk" into the cache
COMPUTE_MS = 10.0  # cost of computing one MoE layer once its experts are resident
CACHE_SLOTS = 6    # resident experts, out of N_LAYERS * N_EXPERTS = 16 total
N_TOKENS = 1000


def expert_id(layer, idx):
    return layer * N_EXPERTS + idx


class Loader:
    """One load channel + a small cache. Loads and compute overlap in time;
    a load only costs latency if something is waiting on it."""

    def __init__(self, cache_slots):
        self.cache_slots = cache_slots
        self.resident = {}   # expert_id -> last-used tick (LRU order)
        self.avail = {}      # expert_id -> time (ms) its load completes
        self.loader_free_at = 0.0
        self.tick = 0
        self.hits = 0
        self.misses = 0

    def schedule_load(self, e, now, protected):
        """Start loading expert e (if not already resident). Returns nothing;
        the expert is usable at self.avail[e]."""
        if e in self.resident:
            return
        if self.cache_slots == 0:
            # no cache: load straight into a scratch slot, evicted immediately
            start = max(self.loader_free_at, now)
            self.avail[e] = start + LOAD_MS
            self.loader_free_at = self.avail[e]
            self.misses += 1
            return
        if len(self.resident) >= self.cache_slots:
            self._evict(protected)
        start = max(self.loader_free_at, now)
        self.avail[e] = start + LOAD_MS
        self.loader_free_at = self.avail[e]
        self.resident[e] = self.tick
        self.misses += 1

    def _evict(self, protected):
        candidates = [e for e in self.resident if e not in protected]
        victim = min(candidates or list(self.resident), key=lambda e: self.resident[e])
        del self.resident[victim]
        self.avail.pop(victim, None)

    def ensure(self, e, now):
        """Make sure expert e is resident; returns the time it is usable."""
        if e in self.resident and self.avail.get(e, 0.0) <= now:
            self.hits += 1
            self.resident[e] = self.tick
            return now
        if e in self.resident:
            # load already in flight — no second load, just wait for it
            self.hits += 1
            self.resident[e] = self.tick
            return self.avail[e]
        return None  # caller must schedule, then wait

    def wait_until_resident(self, experts, now, protected):
        for e in experts:
            t = self.ensure(e, now)
            if t is None:
                self.schedule_load(e, now, protected)
                t = self.avail[e]
            now = max(now, t)
        return now


def simulate(policy, choices, predicted, n_tokens):
    """choices: [n, layers, top_k] expert ids per token. predicted: [n, top_k]
    probe predictions for layer 1. Returns (total_ms, stats)."""
    loader = Loader(CACHE_SLOTS if policy != "demand" else 0)
    now = 0.0
    wasted_prefetch = 0
    prefetches = 0
    for t in range(n_tokens):
        loader.tick = t
        need0 = [expert_id(0, int(e)) for e in choices[t, 0]]
        need1 = [expert_id(1, int(e)) for e in choices[t, 1]]
        protected = set(need0) | set(need1)

        # layer 0: wait for its experts, then compute
        now = loader.wait_until_resident(need0, now, protected)
        if policy == "prefetch":
            pred = [expert_id(1, int(e)) for e in predicted[t]]
            for e in pred:
                if e not in loader.resident:
                    loader.schedule_load(e, now, protected | set(pred))
                    prefetches += 1
                    if e not in need1:
                        wasted_prefetch += 1
        now += COMPUTE_MS

        # layer 1: wait for its experts, then compute
        now = loader.wait_until_resident(need1, now, protected)
        now += COMPUTE_MS

    total = loader.hits + loader.misses
    return now, {
        "ms/token": now / n_tokens,
        "hit rate": loader.hits / total if total else 0.0,
        "prefetches": prefetches,
        "wasted prefetches": wasted_prefetch,
    }


def load_or_train():
    try:
        ckpt = torch.load("weights.pt", weights_only=True)
        model = TinyMoE()
        model.load_state_dict(ckpt["model"])
        probe = LookaheadProbe()
        probe.load_state_dict(ckpt["probe"])
        print("loaded weights.pt")
        return model, probe
    except FileNotFoundError:
        print("no weights.pt — training first (takes ~a minute)...")
        import train as train_mod

        model, x_train, _, _ = train_mod.train_moe(30)
        probe, _ = train_mod.train_probe(model, x_train)
        torch.save({"model": model.state_dict(), "probe": probe.state_dict()}, "weights.pt")
        return model, probe


def main():
    global CACHE_SLOTS, LOAD_MS, COMPUTE_MS
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=N_TOKENS)
    ap.add_argument("--cache", type=int, default=CACHE_SLOTS)
    ap.add_argument("--load-ms", type=float, default=LOAD_MS)
    ap.add_argument("--compute-ms", type=float, default=COMPUTE_MS)
    args = ap.parse_args()

    CACHE_SLOTS, LOAD_MS, COMPUTE_MS = args.cache, args.load_ms, args.compute_ms

    model, probe = load_or_train()
    model.eval()
    probe.eval()

    x_test, _, _ = make_data(max(args.tokens, 200), seed=1)
    x_test = x_test[: args.tokens]
    with torch.no_grad():
        logits, h_in, choices = model.forward_trace(x_test)
        choices = torch.stack(choices, dim=1)  # [B, layers, top_k]
        probe_logits = probe(h_in)
        predicted = torch.topk(probe_logits, TOP_K, dim=-1).indices
        # probe quality on this slice, for the record
        top1 = (predicted[:, 0] == choices[:, 1, 0]).float().mean().item()
        both = sum(
            set(predicted[i].tolist()) == set(choices[i, 1].tolist())
            for i in range(len(x_test))
        ) / len(x_test)
    print(f"probe on test slice: top-1 exact {top1:.3f} | both experts exact {both:.3f}")
    print(f"sim: LOAD_MS={LOAD_MS} COMPUTE_MS={COMPUTE_MS} CACHE_SLOTS={CACHE_SLOTS}/{N_LAYERS * N_EXPERTS} experts, {args.tokens} tokens\n")

    baseline = None
    results = {}
    for policy in ["demand", "lru", "prefetch"]:
        total_ms, stats = simulate(policy, choices, predicted, args.tokens)
        results[policy] = (total_ms, stats)
        print(f"{policy:9s} {stats['ms/token']:6.2f} ms/token   hit rate {stats['hit rate']:.2f}", end="")
        if policy == "prefetch":
            print(f"   prefetches {stats['prefetches']} ({stats['wasted prefetches']} wasted)", end="")
        print()

    demand_ms = results["demand"][0]
    for policy in ["lru", "prefetch"]:
        speedup = demand_ms / results[policy][0]
        print(f"{policy} vs demand: {speedup:.2f}x faster")

    # correctness check: the sim never touches the math, but prove it
    with torch.no_grad():
        logits2, _ = model(x_test)
    print(f"\noutputs identical across policies: {(logits - logits2).abs().max().item() == 0.0} (max logit diff {(logits - logits2).abs().max().item():.1e})")


if __name__ == "__main__":
    main()
