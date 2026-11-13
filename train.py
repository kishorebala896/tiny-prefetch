"""Train the tiny MoE, then train the lookahead probe on its frozen activations.

Usage:
    python train.py              # ~30-60s on CPU
    python train.py --epochs 50

Writes weights.pt (model + probe, local only — checkpoints stay out of the repo).
"""

import argparse

import torch
import torch.nn.functional as F

from moe import N_EXPERTS, TOP_K, LookaheadProbe, TinyMoE, make_data

DEVICE = "cpu"


def train_moe(epochs=30, seed=0):
    torch.manual_seed(seed)
    model = TinyMoE().to(DEVICE)
    x_train, y_train, _ = make_data(6000, seed=seed)
    x_test, y_test, _ = make_data(1200, seed=seed + 1)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        tot_loss = tot_acc = 0.0
        for i in range(0, len(x_train), 256):
            idx = perm[i : i + 256]
            logits, aux = model(x_train[idx])
            loss = F.cross_entropy(logits, y_train[idx]) + 0.01 * aux
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            tot_acc += (logits.argmax(-1) == y_train[idx]).float().mean().item() * len(idx)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:2d}  loss {tot_loss / len(x_train):.4f}  train acc {tot_acc / len(x_train):.3f}")

    model.eval()
    with torch.no_grad():
        logits, _ = model(x_test)
        test_acc = (logits.argmax(-1) == y_test).float().mean().item()
    print(f"test acc: {test_acc:.3f}")
    return model, x_train, x_test, y_test


def train_probe(model, x_train, epochs=300):
    """Fit the probe: h entering layer 0 -> layer 1's top-2 experts.
    Target is a multi-hot vector over the 8 experts."""
    model.eval()
    with torch.no_grad():
        _, h_in, choices = model.forward_trace(x_train)
        target = torch.zeros(len(x_train), N_EXPERTS)
        target.scatter_(1, choices[1], 1.0)  # layer 1's chosen experts

    probe = LookaheadProbe()
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for step in range(epochs):
        idx = torch.randperm(len(x_train))[:1024]
        logits = probe(h_in[idx])
        loss = F.binary_cross_entropy_with_logits(logits, target[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = probe(h_in)
        top2 = torch.topk(logits, TOP_K, dim=-1).indices
        top1_hit = (top2[:, 0] == choices[1][:, 0]).float().mean().item()
        # both predicted experts in the true top-2 (order-free)
        both = torch.tensor([
            set(top2[i].tolist()) == set(choices[1][i].tolist()) for i in range(len(x_train))
        ]).float().mean().item()
        top1_in_top2 = torch.tensor([
            choices[1][i, 0].item() in top2[i].tolist() for i in range(len(x_train))
        ]).float().mean().item()
    print(f"probe: top-1 exact {top1_hit:.3f} | top-1 inside predicted top-2 {top1_in_top2:.3f} | both experts exact {both:.3f}")
    return probe, (top1_hit, top1_in_top2, both)


def routing_table(model, x):
    model.eval()
    with torch.no_grad():
        _, _, choices = model.forward_trace(x)
    print("layer-1 routing by task (rows sum to 1, top expert per task in bold):")
    _, _, task = make_data(len(x), seed=1)
    for t in range(4):
        counts = torch.zeros(N_EXPERTS)
        for e in choices[1][task == t].flatten():
            counts[e] += 1
        counts /= counts.sum()
        print(f"  task {t}: " + " ".join(f"{c:.2f}" for c in counts.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="weights.pt")
    args = ap.parse_args()

    model, x_train, x_test, y_test = train_moe(args.epochs)
    probe, probe_stats = train_probe(model, x_train)
    routing_table(model, x_test)
    torch.save({"model": model.state_dict(), "probe": probe.state_dict()}, args.out)
    print(f"saved {args.out} (local only — not uploaded to the repo)")


if __name__ == "__main__":
    main()
