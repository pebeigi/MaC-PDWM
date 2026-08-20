"""Extra result figures: channel across scenes, closed-loop bars, paired CIs.

Does not overwrite channel.pdf / curves.pdf. Writes to --out (default
paper/figures/results_extra).

    python -m mac.plot_results --out paper/figures/results_extra
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mac.plot_curves import STYLE, collect

ARM_ORDER = ("none", "geometry", "kernel", "history", "diffusion")
SCENE_LABEL = {"cross": "Crossing", "merge": "Merge", "roundabout": "Roundabout"}
PROBE_NAME = {-4.0: "yield", 0.0: "hold", 3.0: "assert"}


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    png = os.path.splitext(path)[0] + ".png"
    fig.savefig(png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def _planner_finals(directory, belief):
    vals = []
    for path in glob.glob(os.path.join(directory, f"metrics_{belief}_*.json")):
        blob = json.load(open(path))
        hist = blob.get("history") or []
        if hist:
            vals.append(hist[-1])
    return vals


def _mean_sd(rows, key, scale=1.0):
    v = np.array([r[key] * scale for r in rows], dtype=float)
    return float(v.mean()), float(v.std())


def plot_channel_scenes(out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.15), sharey=True)
    for ax, scene in zip(axes, ("cross", "merge", "roundabout")):
        ch = json.load(open(f"data/mac/wm_eval_{scene}.json"))["channel"]
        accels = np.asarray(ch["probe_accels"], float)
        ax.plot(accels, ch["p_yield_truth"], "o-", color="#444444",
                label="ground-truth kernel", lw=1.8, markersize=6)
        ax.plot(accels, ch["p_yield_model"], "s-", color="#1b6ca8",
                label="model (deciding)", lw=1.8, markersize=6)
        ax.plot(accels, ch["p_yield_model_all"], "^--", color="#d95f02",
                label="model (all present)", lw=1.5, markersize=5.5)
        ax.set_xticks(accels)
        ax.set_xticklabels([f"{PROBE_NAME.get(a, a)}\n$a={a:+.0f}$" for a in accels])
        ax.set_title(
            f"{SCENE_LABEL[scene]}\n"
            rf"TV truth {ch['tv_truth']:.2f} / model {ch['tv_model']:.2f}",
            fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel(r"$\Pr(\mathrm{yield})$")
    axes[0].legend(fontsize=7, framealpha=0.92, loc="upper left")
    fig.suptitle("Channel recovery across maps", fontsize=11, y=1.03)
    _save(fig, os.path.join(out_dir, "channel_scenes.pdf"))


def plot_planner_return(out_dir):
    dirs = {
        "cross": "data/mac/planner_cross",
        "merge": "data/mac/planner_merge",
        "roundabout": "data/mac/planner_roundabout",
    }
    scenes = list(dirs)
    x = np.arange(len(scenes))
    width = 0.15
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.4))
    for ax, key, scale, ylab, title in (
        (axes[0], "return", 1.0, "Return", "Closed-loop return"),
        (axes[1], "success_rate", 100.0, "Success (%)", "Success rate"),
    ):
        for i, arm in enumerate(ARM_ORDER):
            means, sds = [], []
            for scene in scenes:
                rows = _planner_finals(dirs[scene], arm)
                m, s = _mean_sd(rows, key, scale) if rows else (np.nan, 0.0)
                means.append(m)
                sds.append(s)
            colour, _, label = STYLE[arm]
            ax.bar(x + (i - 2) * width, means, width, yerr=sds, capsize=2.2,
                   color=colour, edgecolor="none", label=label, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENE_LABEL[s] for s in scenes])
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)
    axes[0].legend(fontsize=6.8, ncol=1, framealpha=0.92, loc="upper left")
    _save(fig, os.path.join(out_dir, "planner_return.pdf"))


def plot_paired_delta(out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.2), sharex=True)
    arms = ("kernel", "history", "diffusion")
    y = np.arange(len(arms))
    for ax, scene in zip(axes, ("cross", "merge", "roundabout")):
        paired = json.load(open(f"data/mac/paired_ci_{scene}.json"))["paired"]
        for i, arm in enumerate(arms):
            block = paired[arm]["return"]
            lo, hi = block["ci95"]
            mean = block["mean"]
            colour = STYLE[arm][0]
            ax.plot([lo, hi], [i, i], color=colour, lw=2.4, solid_capstyle="round")
            ax.plot(mean, i, "o", color=colour, markersize=7, zorder=4)
        ax.axvline(0.0, color="#444", lw=0.8, ls="--", zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([STYLE[a][2] for a in arms] if scene == "cross" else [""] * 3)
        ax.set_title(SCENE_LABEL[scene], fontsize=10)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        ax.set_xlabel(r"$\Delta$ return vs geometry")
    fig.suptitle(r"Paired $95\%$ CI of return versus geometry", fontsize=11, y=1.03)
    _save(fig, os.path.join(out_dir, "paired_delta.pdf"))


def plot_beta_onoff(out_dir):
    groups = [
        (r"$\beta_2=0$", "data/mac/planner_nochannel",
         ("none", "history", "diffusion")),
        (r"$\beta_2=2.5$", "data/mac/planner_cross",
         ("none", "history", "diffusion")),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    x = np.arange(3)
    width = 0.34
    for gi, (title, directory, arms) in enumerate(groups):
        means, sds = [], []
        for arm in arms:
            rows = _planner_finals(directory, arm)
            m, s = _mean_sd(rows, "return")
            means.append(m)
            sds.append(s)
        colour = "#7f7f7f" if gi == 0 else "#1b6ca8"
        ax.bar(x + (gi - 0.5) * width, means, width, yerr=sds, capsize=2.4,
               color=colour, edgecolor="none", label=title, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([STYLE[a][2] for a in groups[0][2]])
    ax.set_ylabel("Return")
    ax.set_title("Channel on vs off (crossing)", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.92)
    ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)
    _save(fig, os.path.join(out_dir, "beta_onoff.pdf"))


def plot_sweep(out_dir):
    by_s = {1: [], 4: [], 8: [], 16: []}
    for path in glob.glob("data/mac/planner_sweep/metrics_*.json"):
        blob = json.load(open(path))
        s = int(blob["config"]["n_samples"])
        if blob.get("history"):
            by_s.setdefault(s, []).append(blob["history"][-1])
    for path in glob.glob("data/mac/planner_cross/metrics_diffusion_*.json"):
        blob = json.load(open(path))
        if blob.get("history"):
            by_s[8].append(blob["history"][-1])
    xs = sorted(by_s)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.15))
    for ax, key, scale, ylab, title in (
        (axes[0], "success_rate", 100.0, "Success (%)", "Success vs sample budget"),
        (axes[1], "return", 1.0, "Return", "Return vs sample budget"),
    ):
        means, sds = [], []
        for s in xs:
            m, sd = _mean_sd(by_s[s], key, scale)
            means.append(m)
            sds.append(sd)
        ax.errorbar(xs, means, yerr=sds, fmt="o-", color="#1b6ca8",
                    lw=1.8, capsize=3, markersize=6)
        ax.set_xticks(xs)
        ax.set_xlabel(r"samples $S$")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
    _save(fig, os.path.join(out_dir, "sweep_S.pdf"))


def plot_cf_shift(out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    scenes = ("cross", "merge", "roundabout")
    x = np.arange(len(scenes))
    width = 0.25
    keys = (("gt_shift", "#444444", "simulator $\\mathrm{do}(u)$"),
            ("diffusion_shift", "#1b6ca8", r"diffusion $p(y\mid h,u)$"),
            ("history_shift", "#d95f02", r"history $p(y\mid h)$"))
    for i, (key, colour, label) in enumerate(keys):
        vals = [json.load(open(f"data/mac/wm_counterfactual_{s}.json"))
                ["summary"][key] for s in scenes]
        ax.bar(x + (i - 1) * width, vals, width, color=colour, label=label,
               edgecolor="none", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_LABEL[s] for s in scenes])
    ax.set_ylabel("mean neighbour shift vs hold (m)")
    ax.set_title("Interventional trajectory shift", fontsize=10)
    ax.legend(fontsize=7.5, framealpha=0.92)
    ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)
    _save(fig, os.path.join(out_dir, "cf_shift.pdf"))


def plot_wm_fde(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.25))
    scenes = ("cross", "merge", "roundabout")
    x = np.arange(len(scenes))
    width = 0.2
    series = [
        ("cv_det", "minFDE", "#888888", "CV (det.)"),
        ("cv_stoch", "minFDE", "#4daf4a", "CV + noise $S{=}8$"),
        ("history", "minFDE", "#d95f02", "history"),
        ("diffusion", "minFDE", "#1b6ca8", "diffusion $T{=}25$"),
    ]
    for ax, metric, title in (
        (axes[0], "minFDE", "minFDE (best of $S$)"),
        (axes[1], "meanFDE", "meanFDE"),
    ):
        for i, (block, _, colour, label) in enumerate(series):
            vals = []
            for scene in scenes:
                wm = json.load(open(f"data/mac/wm_eval_{scene}.json"))
                if block == "diffusion":
                    vals.append(wm["diffusion"]["25"][metric])
                else:
                    vals.append(wm[block][metric])
            ax.bar(x + (i - 1.5) * width, vals, width, color=colour,
                   label=label if ax is axes[0] else None, edgecolor="none",
                   zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENE_LABEL[s] for s in scenes])
        ax.set_ylabel("metres")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)
    axes[0].legend(fontsize=7, framealpha=0.92)
    _save(fig, os.path.join(out_dir, "wm_fde.pdf"))


def plot_curves_scenes(out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), sharey=False)
    for ax, scene in zip(axes, ("cross", "merge", "roundabout")):
        runs = collect("logs", scene)
        for arm in ARM_ORDER:
            series = runs.get(arm)
            if not series:
                continue
            max_it = int(max(s[-1, 0] for s in series))
            grid = np.arange(1, max_it + 1, dtype=float)
            stacked = []
            for s in series:
                y = np.full_like(grid, np.nan)
                inside = (grid >= s[0, 0]) & (grid <= s[-1, 0])
                y[inside] = np.interp(grid[inside], s[:, 0], s[:, 1])
                stacked.append(y)
            stack = np.stack(stacked)
            mean = np.nanmean(stack, axis=0)
            sd = np.nanstd(stack, axis=0)
            colour, dash, label = STYLE[arm]
            ax.plot(grid, mean, dash, color=colour,
                    label=f"{label} ($n$={len(series)})", lw=1.8)
            ax.fill_between(grid, mean - sd, mean + sd, color=colour,
                            alpha=0.15, lw=0)
        ax.set_xlabel("PPO iteration")
        ax.set_ylabel("episode return")
        ax.set_title(SCENE_LABEL[scene], fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=6.5, loc="lower right", framealpha=0.92)
    fig.suptitle("Learning curves (return) on all three maps", fontsize=11, y=1.03)
    _save(fig, os.path.join(out_dir, "curves_scenes.pdf"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper/figures/results_extra")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "figure.dpi": 140,
        "savefig.dpi": 170,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    plot_channel_scenes(args.out)
    plot_planner_return(args.out)
    plot_paired_delta(args.out)
    plot_beta_onoff(args.out)
    plot_sweep(args.out)
    plot_cf_shift(args.out)
    plot_wm_fde(args.out)
    plot_curves_scenes(args.out)


if __name__ == "__main__":
    main()
