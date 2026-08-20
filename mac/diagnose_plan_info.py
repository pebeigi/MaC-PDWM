"""Does the ego plan carry information about neighbour intent, given history?

This is the ceiling test for Q3. Any plan-conditioned model p(y|h,u) can only
beat a history-only model p(y|h) if the logged data satisfy I(theta ; u | h) > 0
on the neighbours that actually decide. We fit two matched classifiers of the
realised yield/contest decision -- one on h, one on (h, u) -- and compare
held-out log-loss and AUC. A tie is a property of the environment, not of the
world model.

    python -m mac.diagnose_plan_info --dataset data/mac/cross.npz
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn


def auc(labels, scores):
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def flatten_pairs(history, plan, types, lags=None):
    """One row per (sample, deciding neighbour): ego+neighbour history, plan, label."""
    n = history.shape[0]
    rows_h, rows_u, rows_y, rows_lag = [], [], [], []
    ego = history[:, :, 0, :].reshape(n, -1)
    plan_flat = plan.reshape(n, -1)
    for k in range(types.shape[1]):
        mask = types[:, k] >= 0
        if not mask.any():
            continue
        nb = history[mask, :, k + 1, :].reshape(int(mask.sum()), -1)
        rows_h.append(np.concatenate([ego[mask], nb], axis=1))
        rows_u.append(plan_flat[mask])
        rows_y.append(types[mask, k])
        rows_lag.append(lags[mask, k] if lags is not None
                        else np.full(int(mask.sum()), -1, dtype=np.int64))
    return (np.concatenate(rows_h), np.concatenate(rows_u),
            np.concatenate(rows_y).astype(np.float32), np.concatenate(rows_lag))


def fit(x_train, y_train, x_val, y_val, hidden=128, epochs=30, device="cpu", seed=0):
    """Train a small classifier; return (val logloss, val AUC, per-row val losses)."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    per_row = nn.BCEWithLogitsLoss(reduction="none")
    xt = torch.from_numpy(x_train).float().to(device)
    yt = torch.from_numpy(y_train).float().to(device)
    xv = torch.from_numpy(x_val).float().to(device)
    yv = torch.from_numpy(y_val).float().to(device)

    best = (float("inf"), float("nan"), None)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss_fn(model(xt[idx]).squeeze(-1), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(xv).squeeze(-1)
            val_loss = float(loss_fn(logits, yv))
            if val_loss < best[0]:
                best = (val_loss, auc(y_val, logits.cpu().numpy()),
                        per_row(logits, yv).cpu().numpy())
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/mac/cross.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max_rows", type=int, default=200000)
    parser.add_argument("--json", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = np.load(args.dataset)
    lag_tr = data["lags_train"] if "lags_train" in data.files else None
    lag_va = data["lags_val"] if "lags_val" in data.files else None
    h_tr, u_tr, y_tr, k_tr = flatten_pairs(
        data["history_train"], data["ego_plan_train"], data["types_train"], lag_tr)
    h_va, u_va, y_va, k_va = flatten_pairs(
        data["history_val"], data["ego_plan_val"], data["types_val"], lag_va)

    if len(h_tr) > args.max_rows:
        idx = np.random.default_rng(0).choice(len(h_tr), args.max_rows, replace=False)
        h_tr, u_tr, y_tr, k_tr = h_tr[idx], u_tr[idx], y_tr[idx], k_tr[idx]

    print(f"deciding neighbour rows: {len(h_tr)} train / {len(h_va)} val")
    print(f"base rate P(yield) = {y_tr.mean():.3f} train / {y_va.mean():.3f} val")

    results, rowwise = {}, {}
    for key, name, xt, xv in (
            ("history", "history only  p(theta|h)", h_tr, h_va),
            ("history_plan", "history+plan  p(theta|h,u)",
             np.concatenate([h_tr, u_tr], 1), np.concatenate([h_va, u_va], 1)),
            ("plan", "plan only     p(theta|u)", u_tr, u_va)):
        loss, area, per_row = fit(xt, y_tr, xv, y_va, epochs=args.epochs, device=args.device)
        results[key] = {"val_logloss": loss, "val_auc": area}
        rowwise[key] = per_row
        print(f"  {name:28s} val logloss {loss:.4f}   AUC {area:.4f}")

    hl = results["history"]["val_logloss"]
    hp = results["history_plan"]["val_logloss"]
    print(f"\nplan adds {hl - hp:+.4f} nats of held-out log-likelihood per decision")
    print("a value near zero means no plan-conditioned model can beat p(y|h) here")

    if k_va is not None and (k_va >= 0).any():
        gain = rowwise["history"] - rowwise["history_plan"]
        print("\nplan gain by decision lag (steps from now to the neighbour's commit):")
        by_lag = {}
        for lo, hi in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10)):
            sel = (k_va >= lo) & (k_va <= hi)
            if not sel.any():
                continue
            by_lag[f"{lo}-{hi}"] = {"n": int(sel.sum()), "gain": float(gain[sel].mean())}
            print(f"  lag {lo}-{hi} steps ({0.4 * lo:.1f}-{0.4 * hi:.1f} s): "
                  f"n={int(sel.sum()):5d}  gain {gain[sel].mean():+.4f} nats")
        results["by_lag"] = by_lag

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
