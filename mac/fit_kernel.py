"""Fit the analytic influence kernel from offline data, not from the simulator.

``mac.train_planner.make_oracle_fn`` reads ``env.drivers`` directly: the true
``beta_*`` coefficients, the true stubbornness scale, the running intent EMA, and
the latent ``types``/``resolved`` decisions. That makes the ``kernel`` arm a
privileged ceiling rather than a baseline, so a learned world model cannot beat
it on intention by construction.

This module fits the same functional form to the same offline dataset the world
model is trained on, using only quantities a deployed system could compute:

    P(yield) = sigmoid(w_margin * margin + w_intent * ema(plan) + bias)

``margin`` is the signed-TTC difference already stored by
``mac.data.build_dataset`` (positions, speeds, and a known conflict point).
``ema(plan)`` seeds an exponential moving average of ego acceleration from the
observed history and rolls it forward under the candidate plan, mirroring the
fact that the channel reads intent from recent motion. The averaging window is
selected on the validation split rather than read from the simulator.

    .venv-mac/bin/python -m mac.fit_kernel --data data/mac/roundabout.npz \
        --out data/mac/kernel_roundabout.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

YIELD, CONTEST = 0, 1
# Windows (seconds) searched for the intent EMA. The simulator's own window is
# never read; it is selected by validation loss like any other hyper-parameter.
WINDOW_GRID = (0.4, 0.8, 1.2, 1.6, 2.4, 3.2)
# Steps of the plan the kernel averages over, matching make_oracle_fn's horizon.
DEFAULT_HORIZON = 10
# Signed TTC differences are unbounded when a vehicle is nearly stopped: the raw
# feature reaches several hundred seconds and a linear logit is then fitted to
# the outliers rather than to the decision boundary. Clipping is applied
# identically at fit and query time.
MARGIN_CLIP = 20.0


def ego_speed_series(history):
    """(B, H) ego speed from the history block."""
    return np.linalg.norm(history[:, :, 0, 2:4], axis=-1)


def plan_speed_series(ego_plan):
    """(B, F) ego speed along the plan."""
    return ego_plan[..., 2]


def seed_ema(history, dt, alpha):
    """EMA of ego acceleration over the observed history, per sample."""
    speed = ego_speed_series(history)
    if speed.shape[1] < 2:
        return np.zeros(speed.shape[0], dtype=np.float64)
    accel = np.diff(speed, axis=1) / dt
    ema = np.zeros(accel.shape[0], dtype=np.float64)
    for i in range(accel.shape[1]):
        ema = (1.0 - alpha) * ema + alpha * accel[:, i]
    return ema


def plan_ema(history, ego_plan, dt, window, horizon=DEFAULT_HORIZON):
    """Mean EMA of ego acceleration over the plan window.

    The commitment step is unknown, so the kernel averages the intent signal
    across the horizon the plan covers -- the same window the world model is
    supervised on.
    """
    alpha = float(np.clip(dt / max(window, dt), 0.0, 1.0))
    ema = seed_ema(history, dt, alpha)
    speed = plan_speed_series(ego_plan)
    ego_last = ego_speed_series(history)[:, -1]
    steps = min(horizon, speed.shape[1])
    prev = ego_last
    total = np.zeros_like(ema)
    for n in range(steps):
        accel = (speed[:, n] - prev) / dt
        ema = (1.0 - alpha) * ema + alpha * accel
        total += ema
        prev = speed[:, n]
    return total / max(steps, 1)


def design_matrix(history, ego_plan, margin, dt, window, horizon):
    """Rows of [margin, ema, 1] for every labelled (sample, neighbour) pair."""
    ema = plan_ema(history, ego_plan, dt, window, horizon)          # (B,)
    n_neighbors = margin.shape[1]
    return np.stack([
        np.clip(margin, -MARGIN_CLIP, MARGIN_CLIP).reshape(-1),
        np.repeat(ema, n_neighbors),
        np.ones(margin.size),
    ], axis=1)


def fit_logistic(x, y, l2=1e-3, iters=200):
    """Newton-Raphson logistic regression. Returns the weight vector."""
    w = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(iters):
        z = x @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = x.T @ (p - y) + l2 * w
        s = np.clip(p * (1.0 - p), 1e-6, None)
        hess = x.T @ (x * s[:, None]) + l2 * np.eye(x.shape[1])
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def scores(x, y, w):
    z = np.clip(x @ w, -30, 30)
    p = 1.0 / (1.0 + np.exp(-z))
    logloss = float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
    acc = float(np.mean((p >= 0.5) == (y >= 0.5)))
    base = max(y.mean(), 1 - y.mean())
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    auc = (float((ranks[y >= 0.5].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
           if n_pos > 0 and n_neg > 0 else float("nan"))
    return {"logloss": logloss, "acc": acc, "majority": float(base), "auc": auc,
            "n": int(len(y))}


def labelled(data, split, dt, window, horizon):
    history = data[f"history_{split}"]
    ego_plan = data[f"ego_plan_{split}"]
    margin = data[f"priority_margin_{split}"]
    types = data[f"types_{split}"]
    x = design_matrix(history, ego_plan, margin, dt, window, horizon)
    labels = types.reshape(-1)
    keep = labels >= 0
    # Target is P(yield); YIELD is class 0.
    return x[keep], (labels[keep] == YIELD).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.4)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--l2", type=float, default=1e-3)
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=True)

    best = None
    print(f"{'window':>7s} {'train_ll':>9s} {'val_ll':>9s} {'val_acc':>8s} "
          f"{'majority':>9s} {'val_auc':>8s}")
    for window in WINDOW_GRID:
        xt, yt = labelled(data, "train", args.dt, window, args.horizon)
        xv, yv = labelled(data, "val", args.dt, window, args.horizon)
        w = fit_logistic(xt, yt, l2=args.l2)
        tr, va = scores(xt, yt, w), scores(xv, yv, w)
        print(f"{window:7.1f} {tr['logloss']:9.4f} {va['logloss']:9.4f} "
              f"{va['acc']:8.3f} {va['majority']:9.3f} {va['auc']:8.3f}")
        if best is None or va["logloss"] < best["val"]["logloss"]:
            best = {"window": float(window), "w": w, "train": tr, "val": va}

    w = best["w"]
    params = {
        "beta_margin": float(w[0]),
        "beta_intent": float(w[1]),
        "beta_bias": float(w[2]),
        "intent_window": best["window"],
        "horizon": int(args.horizon),
        "dt": float(args.dt),
        "train": best["train"],
        "val": best["val"],
        "source": args.data,
    }
    with open(args.out, "w") as handle:
        json.dump(params, handle, indent=2)

    print(f"\nselected window {best['window']:.1f}s")
    print(f"  beta_margin={params['beta_margin']:+.4f} "
          f"beta_intent={params['beta_intent']:+.4f} "
          f"beta_bias={params['beta_bias']:+.4f}")
    print(f"  val logloss={best['val']['logloss']:.4f} "
          f"acc={best['val']['acc']:.3f} (majority {best['val']['majority']:.3f}) "
          f"auc={best['val']['auc']:.3f} n={best['val']['n']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
