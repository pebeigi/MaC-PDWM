"""Learning curves for the planner ablation, aggregated over seeds."""
import argparse
import collections
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LINE = re.compile(
    r"iter\s+(\d+)/\d+\s+return=\s*(-?[\d.]+)\s+succ=([\d.]+)\s+coll=([\d.]+)\s+tout=([\d.]+)")

STYLE = {
    "none": ("#888888", "-", "no world model"),
    "geometry": ("#4daf4a", "-", "ego-plan CV risk"),
    "kernel": ("#e6ab02", "--", "analytic channel"),
    "history": ("#d95f02", "--", "history-only WM"),
    "diffusion": ("#1b6ca8", "-", "plan-conditioned (ours)"),
    "mean": ("#d95f02", ":", "collapsed forecast"),
}


def read_log(path):
    rows = []
    with open(path, errors="ignore") as handle:
        for line in handle:
            m = LINE.search(line)
            if m:
                rows.append([float(x) for x in m.groups()])
    return np.asarray(rows) if rows else None


def collect(log_dir, prefix):
    runs = collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(log_dir, f"{prefix}_*.log"))):
        name = os.path.basename(path)[len(prefix) + 1:-len(".log")]
        arm = name.rsplit("_", 1)[0]
        data = read_log(path)
        if data is not None and len(data) > 5:
            runs[arm].append(data)
    return runs


def smooth(y, k=5):
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode="valid")


def panel(ax, runs, col, ylabel, title):
    for arm in ("none", "geometry", "kernel", "history", "diffusion", "mean"):
        series = runs.get(arm)
        if not series:
            continue
        n = min(len(s) for s in series)
        stack = np.stack([smooth(s[:n, col]) for s in series])
        x = np.arange(stack.shape[1]) + 1
        mean, sd = stack.mean(0), stack.std(0)
        colour, dash, label = STYLE[arm]
        ax.plot(x, mean, dash, color=colour, label=f"{label} ($n$={len(series)})", lw=1.8)
        ax.fill_between(x, mean - sd, mean + sd, color=colour, alpha=0.15, lw=0)
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs/v3")
    parser.add_argument("--prefix", default="main")
    parser.add_argument("--out", default="paper/figures/curves.pdf")
    args = parser.parse_args()

    runs = collect(args.logs, args.prefix)
    if not runs:
        raise SystemExit(f"no logs matching {args.prefix}_*.log in {args.logs}")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    panel(axes[0], runs, 2, "success rate", "Successful crossings")
    panel(axes[1], runs, 3, "collision rate", "Collisions")
    panel(axes[2], runs, 1, "episode return", "Return")
    axes[0].legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out} and {png}")
    for arm, series in runs.items():
        finals = [s[-5:, 2].mean() for s in series]
        print(f"  {arm}: n={len(series)} final success {np.mean(finals):.3f} +- {np.std(finals):.3f}")


if __name__ == "__main__":
    main()
