"""Paired confidence intervals and causal acceptance gates for planner runs."""
import argparse
import glob
import json
import os

import numpy as np


def load(directory, final_k):
    rows = {}
    for path in glob.glob(os.path.join(directory, "metrics_*.json")):
        with open(path) as handle:
            blob = json.load(handle)
        config, history = blob.get("config", {}), blob.get("history", [])
        if not history:
            continue
        belief, seed = config.get("belief"), config.get("seed")
        if belief is None or seed is None:
            continue
        window = history[-final_k:]
        rows[(belief, int(seed))] = {
            key: float(np.mean([
                row[key] for row in window
                if isinstance(row.get(key), (int, float))
                and np.isfinite(row[key])
            ]))
            for key in ("return", "success_rate", "collision_rate",
                        "return_infl", "success_rate_infl")
            if any(isinstance(row.get(key), (int, float))
                   and np.isfinite(row[key]) for row in window)
        }
    return rows


def bootstrap_ci(values, rng, draws=20000):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n": int(len(values)), "mean": float(values.mean()),
        "ci95": [float(lo), float(hi)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", required=True)
    ap.add_argument("--reference", default="geometry")
    ap.add_argument("--arms", default="diffusion,history,kernel")
    ap.add_argument("--final_k", type=int, default=1)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    rows = load(args.directory, args.final_k)
    rng = np.random.default_rng(0)
    result = {"directory": args.directory, "reference": args.reference,
              "final_k": args.final_k, "paired": {}}
    for arm in [value.strip() for value in args.arms.split(",") if value.strip()]:
        seeds = sorted(
            seed for belief, seed in rows
            if belief == arm and (args.reference, seed) in rows)
        result["paired"][arm] = {}
        for metric in ("return", "success_rate", "collision_rate",
                       "return_infl", "success_rate_infl"):
            differences = [
                rows[(arm, seed)][metric] - rows[(args.reference, seed)][metric]
                for seed in seeds
                if metric in rows[(arm, seed)]
                and metric in rows[(args.reference, seed)]
            ]
            result["paired"][arm][metric] = bootstrap_ci(differences, rng)
    print(json.dumps(result, indent=2))
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
