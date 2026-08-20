"""Regenerate numeric macros/tables for the self-contained IEEE paper.

The paper source is ``paper/main.tex`` only (no ``\\input`` of result
fragments). This script prints updated macros and planner tables to stdout /
optional files for copy-paste into ``main.tex``; it does not wire them in.

Usage:
    .venv-mac/bin/python -m mac.report_ieee --out /tmp/mac_paper_nums
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ARM_LABEL = {
    "none": r"\texttt{none} (no world model)",
    "geometry": r"\texttt{geometry} (ego-plan CV risk)",
    "kernel": r"\texttt{kernel} (analytic channel feature)",
    "history": r"\texttt{history} (plan dropped)",
    "diffusion": r"\texttt{diffusion} (plan-conditioned)",
}
ARM_ORDER = ["none", "geometry", "kernel", "history", "diffusion"]


def load_planner(directory, commit_steps=None, final_k=1):
    rows = defaultdict(list)
    configs = {}
    for path in sorted(glob.glob(os.path.join(directory, "metrics_*.json"))):
        with open(path) as handle:
            blob = json.load(handle)
        cfg = blob.get("config", {})
        hist = blob.get("history", [])
        if not hist:
            continue
        if commit_steps is not None and cfg.get("commit_steps") != commit_steps:
            continue
        arm = cfg.get("belief")
        if arm is None:
            continue
        window = hist[-final_k:]
        avg = {}
        keys = set().union(*(set(e) for e in window))
        for key in keys:
            vals = [e[key] for e in window if isinstance(e.get(key), (int, float))]
            avg[key] = float(np.mean(vals)) if vals else None
        rows[arm].append(avg)
        configs[arm] = cfg
    return rows, configs


def mean_std(rows, key, scale=1.0):
    vals = [float(r[key]) * scale for r in rows if r.get(key) is not None]
    if not vals:
        return "---"
    return f"{np.mean(vals):.2f}$\\pm${np.std(vals):.2f}"


def planner_rows(rows):
    lines = []
    for arm in ARM_ORDER:
        if arm not in rows:
            continue
        xs = rows[arm]
        n = len(xs)
        line = (
            f"{ARM_LABEL[arm]} & "
            f"{mean_std(xs, 'success_rate', 100)} & "
            f"{mean_std(xs, 'collision_rate', 100)} & "
            f"{mean_std(xs, 'timeout_rate', 100)} & "
            f"{mean_std(xs, 'return')} & "
            f"{mean_std(xs, 'crossing_steps')} & "
            f"{mean_std(xs, 'induced_brakes')} "
            rf"\\ % n={n}"
        )
        lines.append(line)
    return lines


def fmt_pct(x):
    return f"{100 * x:.1f}"


def worldmodel_macros(wm, prefix=""):
    ch = wm["channel"]
    accels = [float(a) for a in ch["probe_accels"]]
    idx = {a: i for i, a in enumerate(accels)}
    yield_a, hold_a, assert_a = min(accels), 0.0, max(accels)
    best_key = sorted(wm["diffusion"], key=int)[-1]
    best = wm["diffusion"][best_key]
    # Prefer T=25 if present for a middle operating point, else best.
    prefer = "25" if "25" in wm["diffusion"] else best_key
    mid = wm["diffusion"][prefer]

    def g(a):
        return f"{ch['p_yield_truth'][idx[a]]:.3f}"

    def m(a):
        return f"{ch['p_yield_model'][idx[a]]:.3f}"

    def ma(a):
        return f"{ch['p_yield_model_all'][idx[a]]:.3f}"

    rows = {
        f"{prefix}numTrainSamples": f"{wm['n_train']:,}".replace(",", "{,}"),
        f"{prefix}numValSamples": f"{wm['n_val']:,}".replace(",", "{,}"),
        f"{prefix}numTrainEpisodes": f"{wm.get('n_train_episodes', 0):,}".replace(",", "{,}"),
        f"{prefix}numValEpisodes": f"{wm.get('n_val_episodes', 0):,}".replace(",", "{,}"),
        f"{prefix}labelledFrac": f"{100 * wm['labelled_frac']:.0f}\\%",
        f"{prefix}cvMinFDE": f"{wm['cv_det']['minFDE']:.2f}",
        f"{prefix}cvMeanFDE": f"{wm['cv_det']['meanFDE']:.2f}",
        f"{prefix}cvMinADE": f"{wm['cv_det']['minADE']:.2f}",
        f"{prefix}cvMeanADE": f"{wm['cv_det']['meanADE']:.2f}",
        f"{prefix}cvsMinFDE": f"{wm['cv_stoch']['minFDE']:.2f}",
        f"{prefix}cvsMeanFDE": f"{wm['cv_stoch']['meanFDE']:.2f}",
        f"{prefix}cvsMinADE": f"{wm['cv_stoch']['minADE']:.2f}",
        f"{prefix}cvsMeanADE": f"{wm['cv_stoch']['meanADE']:.2f}",
        f"{prefix}wmMinFDE": f"{mid['minFDE']:.2f}",
        f"{prefix}wmMeanFDE": f"{mid['meanFDE']:.2f}",
        f"{prefix}wmMinADE": f"{mid['minADE']:.2f}",
        f"{prefix}wmMeanADE": f"{mid['meanADE']:.2f}",
        f"{prefix}wmT": prefer,
        f"{prefix}residStd": f"{wm['resid_std']:.2f}",
        f"{prefix}intentAcc": f"{wm['intent_acc']:.3f}",
        f"{prefix}intentMajority": f"{wm['intent_majority']:.3f}",
        f"{prefix}intentN": f"{wm['intent_n']:,}".replace(",", "{,}"),
        f"{prefix}probeYieldBrake": g(yield_a),
        f"{prefix}probeYieldHold": g(hold_a),
        f"{prefix}probeYieldAssert": g(assert_a),
        f"{prefix}probeModelBrake": m(yield_a),
        f"{prefix}probeModelHold": m(hold_a),
        f"{prefix}probeModelAssert": m(assert_a),
        f"{prefix}probeModelAllBrake": ma(yield_a),
        f"{prefix}probeModelAllHold": ma(hold_a),
        f"{prefix}probeModelAllAssert": ma(assert_a),
        f"{prefix}probeShiftBrake": f"{ch['shifts'][idx[yield_a]]:.2f}",
        f"{prefix}probeShiftAssert": f"{ch['shifts'][idx[assert_a]]:.2f}",
        f"{prefix}probeSpread": f"{ch['spread']:.2f}",
        f"{prefix}gtTV": f"{ch['tv_truth']:.3f}",
        f"{prefix}tvModelDec": f"{ch['tv_model']:.3f}",
        f"{prefix}tvModelAll": f"{ch.get('tv_model_all', float('nan')):.3f}",
        f"{prefix}assertAccel": f"{assert_a:+.0f}",
    }
    if wm.get("diffusion_zero_plan"):
        z = wm["diffusion_zero_plan"]
        rows[f"{prefix}wmZeroMinFDE"] = f"{z['minFDE']:.2f}"
        rows[f"{prefix}wmZeroMeanFDE"] = f"{z['meanFDE']:.2f}"
        if "minADE" in z:
            rows[f"{prefix}wmZeroMinADE"] = f"{z['minADE']:.2f}"
            rows[f"{prefix}wmZeroMeanADE"] = f"{z['meanADE']:.2f}"
    if wm.get("history"):
        h = wm["history"]
        rows[f"{prefix}histMinFDE"] = f"{h['minFDE']:.2f}"
        rows[f"{prefix}histMeanFDE"] = f"{h['meanFDE']:.2f}"
        if "minADE" in h:
            rows[f"{prefix}histMinADE"] = f"{h['minADE']:.2f}"
            rows[f"{prefix}histMeanADE"] = f"{h['meanADE']:.2f}"
    return rows


def paired_macros(ci, prefix=""):
    rows = {}
    for arm in ("diffusion", "history", "kernel"):
        if arm not in ci.get("paired", {}):
            continue
        p = ci["paired"][arm]
        for metric, short in (
                ("return", "Ret"),
                ("success_rate", "Suc"),
                ("collision_rate", "Col")):
            block = p[metric]
            lo, hi = block["ci95"]
            rows[f"{prefix}{arm}{short}"] = f"{block['mean']:+.2f}"
            rows[f"{prefix}{arm}{short}Lo"] = f"{lo:+.2f}"
            rows[f"{prefix}{arm}{short}Hi"] = f"{hi:+.2f}"
            rows[f"{prefix}{arm}{short}N"] = str(block["n"])
            sig = (lo > 0) or (hi < 0)
            rows[f"{prefix}{arm}{short}Sig"] = r"yes" if sig else r"no"
    return rows


def write_macros(path, mapping):
    lines = [
        "% Auto-generated by mac/report_ieee.py --- do not edit by hand.",
        "% Cross-scenario macros without prefix are the crossing defaults.",
        "",
    ]
    for k, v in mapping.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def write_planner_table(path, label, caption, rows):
    body = "\n".join(planner_rows(rows))
    tex = rf"""\begin{{table*}}[!t]
\caption{{{caption}}}
\label{{{label}}}
\centering
\begin{{tabular}}{{|l|c|c|c|c|c|c|}}
\hline
\textbf{{Belief}} & \textbf{{Success (\%)}} & \textbf{{Collision (\%)}} & \textbf{{Timeout (\%)}} & \textbf{{Return}} & \textbf{{Crossing steps}} & \textbf{{Induced brakes}} \\
\hline
{body}
\hline
\end{{tabular}}
\end{{table*}}
"""
    with open(path, "w") as handle:
        handle.write(tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/mac_paper_nums",
                    help="directory for optional generated snippets (not input by main.tex)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # All three scenes use the 4 s receding commit (commit_steps=10), matching
    # scripts/run_experiments.sh PLAN_FLAGS. The 8 s (cs=20) runs are archived.
    scenarios = {
        "cross": 10,
        "merge": 10,
        "roundabout": 10,
    }
    macros = {}
    for scen, cs in scenarios.items():
        wm = json.load(open(f"data/mac/wm_eval_{scen}.json"))
        prefix = "" if scen == "cross" else scen[0].upper() + scen[1:]
        # cross -> ""; merge -> "Merge"; roundabout -> "Roundabout"
        if scen == "merge":
            prefix = "Merge"
        elif scen == "roundabout":
            prefix = "Ra"
        else:
            prefix = ""
        macros.update(worldmodel_macros(wm, prefix))
        macros[f"{prefix}numEpisodes"] = (
            f"{wm.get('n_train_episodes', 0) + wm.get('n_val_episodes', 0):,}"
            .replace(",", "{,}"))

        rows, _ = load_planner(f"data/mac/planner_{scen}", commit_steps=cs)
        for arm in ARM_ORDER:
            if arm not in rows:
                continue
            xs = rows[arm]
            pfx = prefix + arm.capitalize()
            macros[f"{pfx}Return"] = mean_std(xs, "return")
            macros[f"{pfx}Success"] = mean_std(xs, "success_rate", 100)
            macros[f"{pfx}Collision"] = mean_std(xs, "collision_rate", 100)
            macros[f"{pfx}Timeout"] = mean_std(xs, "timeout_rate", 100)
            macros[f"{pfx}N"] = str(len(xs))

        ci_path = f"data/mac/paired_ci_{scen}.json"
        if os.path.exists(ci_path):
            ci = json.load(open(ci_path))
            macros.update(paired_macros(ci, prefix))

        write_planner_table(
            os.path.join(args.out, f"tab_planner_{scen}.tex"),
            f"tab:{'planner' if scen == 'cross' else scen}",
            (
                f"Closed-loop performance on the {scen} scenario "
                f"(commit\\_steps$={cs}$, plan-as-action, $F=20$), "
                r"mean $\pm$ s.d.\ over seeds."
            ),
            rows,
        )

    # Alias main planner macros to crossing for back-compat.
    write_macros(os.path.join(args.out, "results_macros.tex"), macros)
    print(f"wrote {args.out}/results_macros.tex and tab_planner_*.tex")
    print("key macros:")
    for k in ("wmMinFDE", "tvModelDec", "gtTV", "DiffusionReturn",
              "MergeDiffusionReturn", "diffusionRet", "MergediffusionRet"):
        if k in macros:
            print(f"  {k} = {macros[k]}")
    # print merge diffusion delta
    for k, v in macros.items():
        if "diffusionRet" in k or "DiffusionReturn" in k:
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
