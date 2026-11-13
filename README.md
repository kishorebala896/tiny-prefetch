# tiny-prefetch

Yesterday (Nov 12) Edge0 open-sourced a 145B-parameter MoE that they say runs on a single laptop — 29GB peak memory, 80+ tokens a second. I don't have a 145B model lying around to check that, and honestly the benchmark numbers are theirs, not independently verified. But the mechanism behind the claim is interesting on its own: the model is a latent sparse MoE, and the serving stack predicts which experts the next layer will need and loads them *before* they're asked for, so the disk fetch hides inside the compute that's already happening.

That's the part I wanted to see up close. Not the 145B weights — the prefetching trick, at a size where I can actually watch it work.

So this is a 35K-parameter MoE, two layers, eight experts each, trained on a synthetic four-task problem in about forty seconds on CPU. Then a one-layer linear probe that looks at the hidden state going into layer 0 and guesses which experts layer 1 will pick. Then a little discrete-event simulator that serves the same 1,000 tokens three ways — load everything on demand, LRU cache, and cache plus probe-driven prefetch — and reports what each policy actually costs.

## What it does

```bash
pip install -r requirements.txt
python train.py    # trains the MoE (~40s CPU) and the probe, saves weights.pt
python serve.py    # the simulation: demand vs LRU vs prefetch
```

`serve.py` loads `weights.pt` if it's there, otherwise it trains first, so a fresh clone just works. There are flags if you want to play with the assumptions: `--cache 10`, `--load-ms 12`, `--tokens 2000`.

## What I got

The model itself is unremarkable — 96% test accuracy on the toy tasks, which is what you'd expect. The probe is the interesting bit: it gets layer 1's top expert right 70% of the time from the hidden state entering layer 0, and both experts right 60% of the time. The top expert is inside its predicted top-2 about 91% of the time. So prediction is real but genuinely imperfect, which is exactly what makes the serving numbers worth looking at.

The simulation assumes a 6ms disk load per expert, 10ms of compute per layer, and a cache that holds 6 of the 16 experts. One loader, loads overlap compute:

```
demand     44.00 ms/token   hit rate 0.00
lru        34.87 ms/token   hit rate 0.38
prefetch   30.05 ms/token   hit rate 0.47   prefetches 1151 (256 wasted)
lru vs demand: 1.26x faster
prefetch vs demand: 1.46x faster
```

Prefetching beats plain LRU by about 14% here. Not a revolution — a real, measured edge, and it comes with a cost you can count: 256 of the 1,151 prefetches (22%) loaded an expert nobody ended up using. Those wasted loads eat loader time and evict things that might have been useful. The win exists *because* the probe is right often enough, and it shrinks exactly as fast as the probe gets worse. That relationship — predictor quality as the throttle on the whole serving win — is the thing I'd actually take into a real system.

Two ablations worth knowing:

- Bigger cache (`--cache 10`): prefetch still wins, 25.66 vs 28.62 ms/token. The cache helps everyone; prefetch's edge is about *timing*, not just hits.
- Slower disk (`--load-ms 12`): prefetch 43.63 vs LRU 49.56. The slower the storage, the more hiding the load is worth — which is presumably why this matters most on a laptop with a big model and a slow disk, not in a datacenter with everything in HBM.

One thing I checked explicitly: prefetching never changes the outputs. The sim only moves *when* weights arrive, never *which* weights compute. Verified — max logit difference across policies is exactly 0.0. If your prefetching scheme changes your outputs, something is very wrong.

## How the pieces fit

`moe.py` is the model and the data. Four tasks, each with its own labeling rule on eight continuous features, so the router has something real to specialize on. Two MoE layers, top-2 routing, a small load-balancing loss. Nothing fancy — the routing table in the training output shows the specialization anyway.

`train.py` trains the model, then freezes it and trains the probe. The probe's input is the hidden state entering layer 0; its target is layer 1's chosen experts as a multi-hot vector. One linear layer, binary cross-entropy, done. I keep being surprised how far one linear layer gets you when the signal is actually there.

`serve.py` is the simulator. One load channel, one cache, a clock. Each token waits for its layer-0 experts, computes, then waits for layer-1 experts and computes. In the prefetch policy, the probe fires as soon as layer 0 starts computing, and the loader fetches the predicted layer-1 experts while that compute is running — a 6ms load fits inside a 10ms compute window, so a correct prediction costs nothing. A wrong one costs a load and a cache slot, and the sim counts both.

## Rough edges

- The latencies are made up. 6ms loads and 10ms layers are plausible-shaped numbers, not measurements — I picked them so the tradeoff is visible, not so the absolute ms/token means anything. The ratios are the point, and even those only hold while the assumptions do.
- The probe cheats a little, in the sense that routing in this toy is partly task-determined and the task id is sitting right there in the input. At real scale the signal is noisier and the probe would need to be a real model, not one linear layer. The 70% top-1 number is a property of this toy, not a claim about MoE serving in general.
- Tokens are served strictly one at a time. Real serving batches, and batching changes everything about which experts are hot. A batched version of this sim would be the honest next step.
- The cache is per-device and the loader is single-channel. Real stacks have async multi-stream loading, which would make prefetching look better than it does here — or worse, if the loader contends with compute for bandwidth. I didn't model bandwidth contention at all.

## TODO

- Batch the sim (even a batch of 8) and see how the prefetch win holds up when experts are shared across tokens.
- Give the probe the layer-0 router logits as an extra input — right now it only sees the hidden state, and the router's own opinion is free information.
- Try a probe that predicts two layers ahead, or one that predicts the *next token's* layer-0 experts from the current token. Same idea, longer horizon, probably worse accuracy — worth measuring rather than assuming.
- A `--real-disk` mode that actually pages expert weights off a slow device instead of simulating the cost. The sim is honest about being a sim, but there's a difference between modelling a load and feeling one.

## Why this week

Edge0's release is the freshest MoE-serving story around, and it landed the day before I built this. The 145B-on-a-laptop headline is their claim, not mine — but the mechanism underneath it, predict-then-prefetch, is small enough to build from scratch in an evening, and it generalizes: any time your model has a predictable access pattern and a slow tier of storage, a decent predictor turns latency into throughput. That's a useful thing to have actually built once, instead of just nodding along when someone mentions expert caching in a design doc.
