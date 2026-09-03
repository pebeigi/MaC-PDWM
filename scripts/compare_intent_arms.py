#!/usr/bin/env python3
"""Score the world model and the offline-fitted kernel on identical samples.

``mac.eval_world_model`` reports intent accuracy over the whole validation
split, which mixes observational episodes (where the plan is confounded with
the history, so p(y|h,u) is not the interventional quantity either predictor is
meant to answer) with open-loop episodes where it is identifiable. Comparing a
plan-conditioned model against the kernel on the pooled split therefore
understates both. This scores every predictor on the same rows, split out by
whether the episode was interventional.

    PYTHONPATH=. .venv-mac/bin/python scripts/compare_intent_arms.py \
        --dataset data/mac/roundabout.npz \
        --checkpoint data/mac/world_model_roundabout.pt \
        --history_checkpoint data/mac/world_model_history_roundabout.pt \
        --kernel data/mac/kernel_roundabout.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mac.fit_kernel import MARGIN_CLIP, design_matrix
from mac.models.diffusion_world_model import DiffusionWorldModel

YIELD = 0


def metrics(p_yield, y):
    """p_yield: predicted P(class 0); y: 1 where the truth is YIELD."""
    p = np.clip(p_yield, 1e-6, 1 - 1e-6)
    acc = float(np.mean((p >= 0.5) == (y >= 0.5)))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos, n_neg = y.sum(), (1 - y).sum()
    auc = (float((ranks[y >= 0.5].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
           if n_pos > 0 and n_neg > 0 else float("nan"))
    return {"n": int(len(y)), "acc": acc, "logloss": logloss, "auc": auc,
            "majority": float(max(y.mean(), 1 - y.mean()))}


def wm_intent(path, data, device, zero_plan=False, batch=512):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = DiffusionWorldModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    pos = ckpt.get("pos_scale", 30.0)
    vel = ckpt.get("vel_scale", 15.0)

    hist = data["history_val"]
    plan = data["ego_plan_val"]
    out = []
    with torch.no_grad():
        for i in range(0, hist.shape[0], batch):
            h = torch.from_numpy(hist[i:i + batch]).float().to(device)
            h[..., :2] /= pos
            h[..., 2:4] /= vel
            u = torch.from_numpy(plan[i:i + batch]).float().to(device)
            u[..., :2] /= pos
            u[..., 2] /= vel
            if zero_plan:
                u = torch.zeros_like(u)
            probs = model.predict_intentions(h, u)
            out.append(probs[..., YIELD].cpu().numpy())
    return np.concatenate(out, axis=0)        # (B, K) P(yield)


def kernel_intent(params, data, dt):
    x = design_matrix(data["history_val"], data["ego_plan_val"],
                      data["priority_margin_val"], dt,
                      params["intent_window"], params.get("horizon", 10))
    w = np.array([params["beta_margin"], params["beta_intent"],
                  params["beta_bias"]], dtype=np.float64)
    z = np.clip(x @ w, -30, 30)
    n_neighbors = data["priority_margin_val"].shape[1]
    return (1.0 / (1.0 + np.exp(-z))).reshape(-1, n_neighbors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--history_checkpoint", default="")
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--dt", type=float, default=0.4)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    params = json.load(open(args.kernel))

    types = data["types_val"]
    interventional = data["interventional_val"]
    labelled = types >= 0
    y = (types == YIELD).astype(np.float64)
    # (B,) episode flag broadcast to (B,K) neighbour slots
    interv = np.repeat(interventional[:, None], types.shape[1], axis=1)

    preds = {
        "diffusion p(y|h,u)": wm_intent(args.checkpoint, data, device),
        "diffusion p(y|h)": wm_intent(args.checkpoint, data, device, zero_plan=True),
        "kernel (offline fit)": kernel_intent(params, data, args.dt),
    }
    if args.history_checkpoint:
        preds["history WM"] = wm_intent(args.history_checkpoint, data, device)

    subsets = {
        "all labelled": labelled,
        "interventional only": labelled & interv,
        "observational only": labelled & ~interv,
    }

    results = {}
    for sub_name, mask in subsets.items():
        print(f"\n===== {sub_name} =====")
        print(f"{'predictor':24s} {'n':>7s} {'acc':>7s} {'logloss':>8s} {'auc':>7s} "
              f"{'majority':>9s}")
        results[sub_name] = {}
        for name, p in preds.items():
            m = metrics(p[mask], y[mask])
            results[sub_name][name] = m
            print(f"{name:24s} {m['n']:7d} {m['acc']:7.4f} {m['logloss']:8.4f} "
                  f"{m['auc']:7.4f} {m['majority']:9.4f}")

    print("\n=== diffusion minus offline-fitted kernel ===")
    for sub_name in subsets:
        d = results[sub_name]["diffusion p(y|h,u)"]
        k = results[sub_name]["kernel (offline fit)"]
        print(f"  {sub_name:22s} acc {d['acc'] - k['acc']:+.4f}  "
              f"auc {d['auc'] - k['auc']:+.4f}  "
              f"logloss {d['logloss'] - k['logloss']:+.4f} (lower is better)")

    print("\n=== plan-conditioning gain (p(y|h,u) minus p(y|h)) ===")
    for sub_name in subsets:
        d = results[sub_name]["diffusion p(y|h,u)"]
        z = results[sub_name]["diffusion p(y|h)"]
        print(f"  {sub_name:22s} acc {d['acc'] - z['acc']:+.4f}  "
              f"auc {d['auc'] - z['auc']:+.4f}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
