"""Build the paper's LaTeX constants and tables from run artefacts.

Everything quoted in the paper is generated here, so the text cannot drift away
from the logs. Run after the world-model evaluation and the planner sweeps:

    python -m mac.report --wm data/mac/wm_eval.json \\
        --main data/mac/planner_main --nochannel data/mac/planner_nochannel \\
        --sweep data/mac/planner_sweep --out paper
"""
import argparse
import glob
import json
import os

import numpy as np

ARM_LABEL = {
    "none": r"\texttt{none} (no world model)",
    "geometry": r"\texttt{geometry} (ego-plan CV risk)",
    "kernel": r"\texttt{kernel} (analytic channel feature)",
    "mean": r"\texttt{mean} (collapsed forecast)",
    "diffusion": r"\texttt{diffusion} (multi-hypothesis)",
    "history": r"\texttt{history} (plan dropped)",
    "gap": r"\texttt{gap} (time-gap acceptance)",
    "constant": r"\texttt{constant} (always-go)",
}
ARM_ORDER = ["none", "geometry", "kernel", "mean", "history", "diffusion"]


FINAL_K = 3


def load_runs(directory, final_k=FINAL_K):
    """Return {run_name: (config, metrics)} for every finished run.

    Metrics are averaged over the last ``final_k`` evaluation points rather than
    read off the final one. A single evaluation of a PPO policy is a noisy
    estimate of that run's quality, and late-training policies drift; averaging a
    fixed, pre-specified window is the standard remedy. The final-iterate numbers
    are reported alongside in the text so the difference is visible.
    """
    runs = {}
    for path in sorted(glob.glob(os.path.join(directory, "metrics_*.json"))):
        with open(path) as handle:
            blob = json.load(handle)
        if isinstance(blob, list):            # older format
            config, history = {}, blob
        else:
            config, history = blob.get("config", {}), blob.get("history", [])
        if not history:
            continue
        window = history[-final_k:]
        keys = set().union(*(set(e) for e in window))
        averaged = {}
        for key in keys:
            values = [e[key] for e in window
                      if isinstance(e.get(key), (int, float))
                      and not (isinstance(e[key], float) and np.isnan(e[key]))]
            averaged[key] = float(np.mean(values)) if values else None
        name = os.path.basename(path)[len("metrics_"):-len(".json")]
        runs[name] = (config, averaged)
    return runs


# Settings that change what a run means. Two runs that differ on any of these do
# not belong in the same table row, and averaging them silently produces a number
# that describes no actual experiment.
ROW_CRITICAL_FIELDS = ("scenario", "time_penalty", "beta_intent", "beta_margin",
                       "type_probs", "decision_distance", "intent_window",
                       "bg_scale", "iterations", "commit_steps", "plan_action")


def group(runs, key):
    """Group final metrics by a config field."""
    out = {}
    configs = {}
    for name, (config, metrics) in runs.items():
        bucket = config.get(key)
        out.setdefault(bucket, []).append(metrics)
        configs.setdefault(bucket, []).append((name, config))
    for bucket, entries in configs.items():
        for field in ROW_CRITICAL_FIELDS:
            seen = {repr(cfg.get(field)) for _, cfg in entries if field in cfg}
            if len(seen) > 1:
                print(f"WARNING: {key}={bucket!r} mixes {field}={sorted(seen)} "
                      f"across {len(entries)} runs; that row averages different "
                      f"experiments", flush=True)
    return out


def agg(entries, field, scale=1.0):
    values = [e[field] * scale for e in entries if e.get(field) is not None
              and not (isinstance(e[field], float) and np.isnan(e[field]))]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def fmt(mean, std, digits=1):
    if mean is None:
        return "--"
    return f"${mean:.{digits}f} \\pm {std:.{digits}f}$"


def planner_table(runs, caption, label, arms=ARM_ORDER):
    by_arm = group(runs, "belief")
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Belief & Success (\%) & Collision (\%) & Timeout (\%) "
        r"& Return & Crossing steps & Induced brakes \\",
        r"\midrule",
    ]
    for arm in arms:
        entries = by_arm.get(arm)
        if not entries:
            continue
        n = len(entries)
        row = " & ".join([
            ARM_LABEL.get(arm, arm),
            fmt(*agg(entries, "success_rate", 100.0)),
            fmt(*agg(entries, "collision_rate", 100.0)),
            fmt(*agg(entries, "timeout_rate", 100.0)),
            fmt(*agg(entries, "return"), ),
            fmt(*agg(entries, "crossing_steps")),
            fmt(*agg(entries, "induced_brakes")),
        ])
        lines.append(row + rf" \\ % n={n}")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def beta_table(runs):
    buckets = {}
    for _, (config, metrics) in runs.items():
        beta = config.get("beta_intent")
        belief = config.get("belief")
        if beta is None or belief is None:
            continue
        buckets.setdefault((float(beta), belief), []).append(metrics)
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{Channel-strength sweep. Same world models (trained with "
        r"$\beta_2>0$); only the environment kernel changes. "
        r"\texttt{history} vs \texttt{diffusion} at $\beta_2=0$ is the "
        r"plan-conditioning ablation with the type-channel removed. "
        r"Mean $\pm$ s.d.\ over seeds.}",
        r"\label{tab:beta}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"$\beta_2$ & Belief & Success (\%) & Collision (\%) & Timeout (\%) & Return \\",
        r"\midrule",
    ]
    betas = sorted({b for b, _ in buckets})
    for i, beta in enumerate(betas):
        if i:
            lines.append(r"\midrule")
        for belief in ["none", "history", "diffusion", "mean"]:
            entries = buckets.get((beta, belief))
            if not entries:
                continue
            row = " & ".join([
                f"${beta:.2f}$",
                ARM_LABEL.get(belief, belief),
                fmt(*agg(entries, "success_rate", 100.0)),
                fmt(*agg(entries, "collision_rate", 100.0)),
                fmt(*agg(entries, "timeout_rate", 100.0)),
                fmt(*agg(entries, "return")),
            ])
            lines.append(row + rf" \\ % n={len(entries)}")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def sweep_table(runs):
    by_s = group(runs, "n_samples")
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\small",
        r"\caption{Sample budget $S$ for the belief features. "
        r"Remark~\ref{rem:hoeffding} bounds the $95\%$ resolution of the risk "
        r"estimate by $\epsilon \approx \sqrt{\ln(40)/(2S)}$, shown for reference. "
        r"Mean $\pm$ s.d. over seeds.}",
        r"\label{tab:sweep}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"$S$ & $\epsilon_{95}$ & Success (\%) & Collision (\%) & Timeout (\%) & Return \\",
        r"\midrule",
    ]
    for s in sorted(k for k in by_s if k is not None):
        entries = by_s[s]
        eps = np.sqrt(np.log(40.0) / (2.0 * s))
        row = " & ".join([
            f"${s}$", f"${eps:.2f}$",
            fmt(*agg(entries, "success_rate", 100.0)),
            fmt(*agg(entries, "collision_rate", 100.0)),
            fmt(*agg(entries, "timeout_rate", 100.0)),
            fmt(*agg(entries, "return")),
        ])
        lines.append(row + rf" \\ % n={len(entries)}")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def counterfactual_table(cf):
    s = cf["summary"]

    def num(key):
        val = s.get(key)
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return "--"
        return f"{val:.2f}"

    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Interventional counterfactuals. The same scene $h$ is "
        r"replayed under several ego plans $\mathrm{do}(u)$ in the simulator. "
        r"Shift is mean neighbour displacement vs the hold plan. A history-only "
        r"model cannot move its forecast with $u$.}",
        r"\label{tab:cf}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"Model & minFDE & meanFDE & Pred.\ shift (m) & GT shift (m) \\",
        r"\midrule",
        f"Diffusion $p(y\\mid h,u)$ & {num('diffusion_minFDE')} & "
        f"{num('diffusion_meanFDE')} & {num('diffusion_shift')} & "
        f"{num('gt_shift')} " + r"\\",
        f"History $p(y\\mid h)$ & {num('history_minFDE')} & "
        f"{num('history_meanFDE')} & {num('history_shift')} & "
        f"{num('gt_shift')} " + r"\\",
        r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", "",
    ]
    return "\n".join(lines)


def worldmodel_table(wm):
    det, stoch = wm["cv_det"], wm["cv_stoch"]
    steps = sorted(wm["diffusion"], key=int)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Displacement error on held-out \emph{episodes}, in metres, "
        r"$K=5$ neighbours over $F=10$ steps ($4$\,s). minFDE is a best-of-$S$ "
        r"statistic and is only comparable across rows with the same $S$; the "
        r"deterministic constant-velocity roll-out is therefore the reference "
        r"for meanFDE, and the noise-perturbed roll-out with a matched $S=8$ "
        r"budget is the reference for minFDE.}",
        r"\label{tab:worldmodel}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Model & $S$ & minFDE & meanFDE & minADE & meanADE \\",
        r"\midrule",
        f"Constant velocity & 1 & {det['minFDE']:.2f} & {det['meanFDE']:.2f} & "
        f"{det['minADE']:.2f} & {det['meanADE']:.2f} " + r"\\",
        f"Constant velocity $+$ noise & 8 & {stoch['minFDE']:.2f} & {stoch['meanFDE']:.2f} & "
        f"{stoch['minADE']:.2f} & {stoch['meanADE']:.2f} " + r"\\",
        r"\midrule",
    ]
    for s in steps:
        m = wm["diffusion"][s]
        lines.append(
            f"Diffusion $p(y\\mid h,u)$ (DDIM $T={s}$) & 8 & {m['minFDE']:.2f} & {m['meanFDE']:.2f} & "
            f"{m['minADE']:.2f} & {m['meanADE']:.2f} " + r"\\")
    if wm.get("diffusion_zero_plan"):
        z = wm["diffusion_zero_plan"]
        lines.append(
            f"Diffusion $p(y\\mid h)$ (plan zeroed, $T=10$) & 8 & {z['minFDE']:.2f} & "
            f"{z['meanFDE']:.2f} & {z['minADE']:.2f} & {z['meanADE']:.2f} " + r"\\")
    if wm.get("history"):
        h = wm["history"]
        lines.append(
            f"History-only WM $p(y\\mid h)$ ($T=10$) & 8 & {h['minFDE']:.2f} & "
            f"{h['meanFDE']:.2f} & {h['minADE']:.2f} & {h['meanADE']:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]

    ch = wm["channel"]
    names = {-4.0: "yield", -3.0: "yield", 0.0: "hold", 2.0: "assert", 3.0: "assert"}
    lines += [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Channel recovery. The model is probed with the three "
        r"candidate intents; the simulator's kernel requires $\Pr(\yield)$ to "
        r"increase with ego acceleration. Trajectory shift is the mean "
        r"displacement of the predicted futures relative to the \emph{hold} "
        r"probe, to be read against the spread across diffusion samples "
        rf"({ch['spread']:.2f}\,m): a probe effect below that is not usable by "
        r"a planner. Ground truth uses the held-out scene's TTC margin when "
        r"that metadata is available.}",
        r"\label{tab:channel}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"& & \multicolumn{2}{c}{$\Pr(\yield)$ model} & & \\",
        r"\cmidrule(lr){3-4}",
        r"Probe $u$ & $a$ (m/s$^2$) & deciding & all present & $\Pr(\yield)$ truth "
        r"& Traj.\ shift (m) \\",
        r"\midrule",
    ]
    all_p = ch.get("p_yield_model_all", [float("nan")] * len(ch["probe_accels"]))
    # shifts[] are already measured vs the hold probe (a=0) in eval_world_model.
    hold_idx = list(ch["probe_accels"]).index(0.0) if 0.0 in ch["probe_accels"] else None
    for i, a in enumerate(ch["probe_accels"]):
        if hold_idx is not None and i == hold_idx:
            shown = "--"
        else:
            shown = f"{float(ch['shifts'][i]):.2f}"
        lines.append(
            f"\\textit{{{names.get(a, a)}}} & ${a:+.0f}$ & "
            f"{ch['p_yield_model'][i]:.3f} & {all_p[i]:.3f} & "
            f"{ch['p_yield_truth'][i]:.3f} & {shown} " + r"\\")
    lines += [
        r"\midrule",
        f"Total variation & & {ch['tv_model']:.3f} & "
        f"{ch.get('tv_model_all', float('nan')):.3f} & {ch['tv_truth']:.3f} & " + r"\\",
        f"Monotone in $a$ & & {'yes' if ch['monotone'] else 'no'} & no & yes & " + r"\\",
        r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", "",
    ]
    return "\n".join(lines)


def constants(wm):
    ch = wm["channel"]
    accels = [float(a) for a in ch["probe_accels"]]
    idx = {a: i for i, a in enumerate(accels)}
    yield_a = min(accels)
    hold_a = 0.0 if 0.0 in idx else accels[len(accels) // 2]
    assert_a = max(accels)

    def gt(a):
        return f"{ch['p_yield_truth'][idx[a]]:.3f}"
    best = wm["diffusion"][sorted(wm["diffusion"], key=int)[-1]]
    rows = {
        "numEpisodes": f"{wm.get('n_episodes', 0):,}".replace(",", "{,}"),
        "numTrainSamples": f"{wm['n_train']:,}".replace(",", "{,}"),
        "numValSamples": f"{wm['n_val']:,}".replace(",", "{,}"),
        "numTrainEpisodes": f"{wm.get('n_train_episodes', 0):,}".replace(",", "{,}"),
        "numValEpisodes": f"{wm.get('n_val_episodes', 0):,}".replace(",", "{,}"),
        "labelledFrac": f"{100 * wm['labelled_frac']:.0f}\\%",
        "cvMinFDE": f"{wm['cv_det']['minFDE']:.2f}",
        "cvMeanFDE": f"{wm['cv_det']['meanFDE']:.2f}",
        "cvMinADE": f"{wm['cv_det']['minADE']:.2f}",
        "cvMeanADE": f"{wm['cv_det']['meanADE']:.2f}",
        "cvsMinFDE": f"{wm['cv_stoch']['minFDE']:.2f}",
        "cvsMeanFDE": f"{wm['cv_stoch']['meanFDE']:.2f}",
        "cvsMinADE": f"{wm['cv_stoch']['minADE']:.2f}",
        "cvsMeanADE": f"{wm['cv_stoch']['meanADE']:.2f}",
        "wmMinFDE": f"{best['minFDE']:.2f}",
        "wmMeanFDE": f"{best['meanFDE']:.2f}",
        "wmMinADE": f"{best['minADE']:.2f}",
        "wmMeanADE": f"{best['meanADE']:.2f}",
        "residStd": f"{wm['resid_std']:.2f}",
        "intentAcc": f"{wm['intent_acc']:.3f}",
        "intentMajority": f"{wm['intent_majority']:.3f}",
        "intentN": f"{wm['intent_n']:,}".replace(",", "{,}"),
        "probeYieldBrake": gt(yield_a),
        "probeYieldHold": gt(hold_a),
        "probeYieldAssert": gt(assert_a),
        "probeShiftBrake": f"{ch['shifts'][idx[yield_a]]:.2f}",
        "probeShiftAssert": f"{ch['shifts'][idx[assert_a]]:.2f}",
        "probeSpread": f"{ch['spread']:.2f}",
        "gtTV": f"{ch['tv_truth']:.3f}",
        "tvModelDec": f"{ch['tv_model']:.3f}",
        "tvModelAll": f"{ch.get('tv_model_all', float('nan')):.3f}",
    }
    if wm.get("diffusion_zero_plan"):
        z = wm["diffusion_zero_plan"]
        rows["wmZeroMinFDE"] = f"{z['minFDE']:.2f}"
        rows["wmZeroMeanFDE"] = f"{z['meanFDE']:.2f}"
    if wm.get("history"):
        h = wm["history"]
        rows["histMinFDE"] = f"{h['minFDE']:.2f}"
        rows["histMeanFDE"] = f"{h['meanFDE']:.2f}"
    lines = ["% Generated by mac/report.py -- do not edit by hand.", ""]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in rows.items()]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm", default="data/mac/wm_eval.json")
    parser.add_argument("--counterfactual", default="data/mac/wm_counterfactual.json")
    parser.add_argument("--main", default="data/mac/planner_main")
    parser.add_argument("--history", default="data/mac/planner_history")
    parser.add_argument("--nochannel", default="data/mac/planner_nochannel_transferWM")
    parser.add_argument("--sweep", default="data/mac/planner_sweep")
    parser.add_argument("--merge", default="data/mac/planner_merge")
    parser.add_argument("--hard", default="data/mac/planner_hard")
    parser.add_argument("--beta", default="data/mac/planner_beta")
    parser.add_argument("--shift", default="data/mac/planner_shift")
    parser.add_argument("--baselines", default="data/mac/baselines")
    parser.add_argument("--geometry", default="data/mac/planner_geometry")
    parser.add_argument("--independent", default="data/mac/planner_independent")
    parser.add_argument("--commit", default="data/mac/planner_commit")
    parser.add_argument("--negotiate", default="data/mac/planner_negotiate")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--out", default="paper")
    args = parser.parse_args()

    with open(args.wm) as handle:
        wm = json.load(handle)
    wm["n_episodes"] = args.episodes or wm.get("n_episodes", 0)

    written = []

    def write(name, text):
        path = os.path.join(args.out, name)
        with open(path, "w") as handle:
            handle.write(text)
        written.append(path)

    write("results.tex", constants(wm))
    # Split so each table can be floated next to the section that discusses it.
    wm_tex = worldmodel_table(wm)
    parts = wm_tex.split(r"\begin{table}[t]")
    # parts[0] is empty/leading whitespace; parts[1] is displacement, parts[2] channel
    write("tab_worldmodel.tex", r"\begin{table}[t]" + parts[1] if len(parts) > 1 else wm_tex)
    if len(parts) > 2:
        write("tab_channel.tex", r"\begin{table}[t]" + parts[2])

    if os.path.isfile(args.counterfactual):
        with open(args.counterfactual) as handle:
            write("tab_cf.tex", counterfactual_table(json.load(handle)))

    if os.path.isdir(args.main):
        runs = load_runs(args.main)
        if os.path.isdir(args.history):
            runs.update(load_runs(args.history))
        write("tab_planner.tex", planner_table(
            runs,
            r"Closed-loop performance at the unsignalised crossing, mean $\pm$ s.d. "
            r"over seeds. Arms differ only in the observation: \texttt{none}, "
            r"\texttt{mean} (collapsed futures), \texttt{history} ($p(y\mid h)$), "
            r"and \texttt{diffusion} ($p(y\mid h,u)$). Crossing steps counts only "
            r"successful episodes.",
            "tab:planner"))

    if os.path.isdir(args.nochannel):
        runs = load_runs(args.nochannel)
        write("tab_influence.tex", planner_table(
            runs,
            r"Type-channel off ($\beta_2 = 0$). Latent types no longer depend "
            r"on ego motion, so Proposition~\ref{prop:influence} forces "
            r"$G_{\mathrm{infl}} = 0$. SUMO car-following still reacts to the "
            r"ego, so a residual gap over \texttt{none} is ordinary prediction "
            r"(including kinematic interaction), not type-negotiation. "
            r"\texttt{history} vs \texttt{diffusion} isolates plan-conditioning "
            r"with the type-channel removed.",
            "tab:influence"))

    if os.path.isdir(args.sweep):
        write("tab_sweep.tex", sweep_table(load_runs(args.sweep)))

    if os.path.isdir(args.merge) and glob.glob(os.path.join(args.merge, "metrics_*.json")):
        write("tab_merge.tex", planner_table(
            load_runs(args.merge),
            r"Closed-loop performance on the merge scenario, mean $\pm$ s.d.\ over "
            r"seeds. Same three belief arms as Table~\ref{tab:planner}.",
            "tab:merge"))

    if os.path.isdir(args.hard) and glob.glob(os.path.join(args.hard, "metrics_*.json")):
        write("tab_hard.tex", planner_table(
            load_runs(args.hard),
            r"Harder crossing: denser traffic, more reactive/contesting drivers, "
            r"and a stronger influence channel. Mean $\pm$ s.d.\ over seeds.",
            "tab:hard"))

    if os.path.isdir(args.beta) and glob.glob(os.path.join(args.beta, "metrics_*.json")):
        write("tab_beta.tex", beta_table(load_runs(args.beta)))

    shift_dir = getattr(args, "shift", "data/mac/planner_shift")
    if os.path.isdir(shift_dir) and glob.glob(os.path.join(shift_dir, "metrics_*.json")):
        write("tab_shift.tex", planner_table(
            load_runs(shift_dir),
            r"Zero-shot distribution shift: policies trained on the default "
            r"crossing, evaluated with denser traffic, a stronger channel, and "
            r"more reactive types. No retraining. Mean $\pm$ s.d.\ over seeds.",
            "tab:shift"))

    if os.path.isdir(args.baselines) and glob.glob(os.path.join(args.baselines, "metrics_*.json")):
        write("tab_baselines.tex", planner_table(
            load_runs(args.baselines),
            r"Rule-based baselines on the same crossing and traffic mix: "
            r"gap-acceptance and always-go. Mean $\pm$ s.d.\ over seeds.",
            "tab:baselines",
            arms=["gap", "constant"]))

    geo_dir = getattr(args, "geometry", "data/mac/planner_geometry")
    if os.path.isdir(geo_dir) and glob.glob(os.path.join(geo_dir, "metrics_*.json")):
        write("tab_geometry.tex", planner_table(
            load_runs(geo_dir),
            r"Geometry-only belief: ego-plan vs constant-velocity neighbour "
            r"roll-outs, no learned world model. Compared with \texttt{none} "
            r"and (when present) the learned arms, this isolates engineered "
            r"conflict-risk features from prediction / influence.",
            "tab:geometry",
            arms=["none", "geometry", "history", "diffusion"]))

    ind_dir = getattr(args, "independent", "data/mac/planner_independent")
    if os.path.isdir(ind_dir) and glob.glob(os.path.join(ind_dir, "metrics_*.json")):
        write("tab_independent.tex", planner_table(
            load_runs(ind_dir),
            r"Independent per-neighbour world model (no joint residual) vs the "
            r"joint diffusion model. Same planner, probes, and seeds.",
            "tab:independent"))

    cmt_dir = getattr(args, "commit", "data/mac/planner_commit")
    if os.path.isdir(cmt_dir) and glob.glob(os.path.join(cmt_dir, "metrics_*.json")):
        write("tab_commit.tex", planner_table(
            load_runs(cmt_dir),
            r"Receding-horizon commit: the chosen acceleration is held for "
            r"several env steps so the executed $u_{t:t+H}$ matches the "
            r"$F$-step plan queried from the world model.",
            "tab:commit"))

    neg_dir = getattr(args, "negotiate", "data/mac/planner_negotiate")
    if os.path.isdir(neg_dir) and glob.glob(os.path.join(neg_dir, "metrics_*.json")):
        write("tab_negotiate.tex", planner_table(
            load_runs(neg_dir),
            r"Plan-as-action: PPO selects among $\{\mathrm{yield},\mathrm{hold},"
            r"\mathrm{assert}\}$ probes and commits that acceleration for $F$ "
            r"steps. All arms share this action space. A diffusion--history gap "
            r"here is evidence the policy uses $p(y\mid h,u)$, not ego geometry.",
            "tab:negotiate",
            arms=["none", "geometry", "history", "diffusion"]))

    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
