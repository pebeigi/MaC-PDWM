"""Evaluate world-model sample quality, intention inference, and channel recovery.

Three things are measured, and the first two are deliberately reported against
baselines that make the comparison fair:

``displacement``  minFDE of a generative model is a best-of-S quantity, so it is
                  compared against a *stochastic* constant-velocity baseline
                  drawing the same number of samples. The deterministic
                  constant-velocity roll-out is reported against meanFDE.
``intention``     accuracy is meaningless without the majority-class rate.
``channel``       whether the learned model reproduces the direction of the
                  simulator's influence kernel when probed counterfactually.
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from mac.data.normalize import POS_SCALE, RESID_SCALE, VEL_SCALE, decode_samples
from mac.data.scene import synthetic_plan
from mac.envs.social_drivers import (BETA_BIAS, BETA_INTENT, BETA_MARGIN,
                                     SocialDriverManager)
from mac.envs.sumo_planning_env import SCENARIOS
from mac.models.belief import DEFAULT_PROBE_ACCELS, parse_probe_accels
from mac.models.diffusion_world_model import DiffusionWorldModel
from mac.train_world_model import load_split


def _fde_ade(pred, target, mask):
    """pred (B,S,F,K,2), target (B,1,F,K,2), mask (B,1,F,K) -> per-agent errors."""
    err = torch.linalg.norm(pred - target, dim=-1)          # (B,S,F,K)
    final_mask = mask[:, 0, -1, :]                          # (B,K)
    fde = err[:, :, -1, :]                                  # (B,S,K)
    steps = mask[:, 0].sum(dim=1).clamp(min=1)              # (B,K)
    ade = (err * mask).sum(dim=2) / steps[:, None, :]       # (B,S,K)
    return fde, ade, final_mask


def _reduce(fde, ade, final_mask):
    """Per-agent min-over-samples and mean-over-samples, averaged over valid agents."""
    valid = final_mask > 0
    n = valid.sum().clamp(min=1)
    out = {}
    for name, value in (("fde", fde), ("ade", ade)):
        out[f"min{name.upper()}"] = float((value.min(dim=1).values * final_mask).sum() / n)
        out[f"mean{name.upper()}"] = float((value.mean(dim=1) * final_mask).sum() / n)
    return out, int(n)


def _score_displacement(model, loader, device, n_samples, steps, batches, zero_plan=False):
    totals, agents = None, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= batches:
                break
            history, ego_plan, _, _, _, history_raw, future_raw = batch
            history, ego_plan = history.to(device), ego_plan.to(device)
            if zero_plan:
                ego_plan = torch.zeros_like(ego_plan)
            history_raw = history_raw.to(device)
            future_raw = future_raw.to(device)
            samples = model.sample(history, ego_plan, n_samples=n_samples, steps=steps)
            pred = decode_samples(samples, history_raw)
            current = history_raw[:, -1, 1:, :2]
            target = (future_raw[..., :2] + current[:, None, :, :]).unsqueeze(1)
            mask = future_raw[..., 2].unsqueeze(1)
            fde, ade, fm = _fde_ade(pred, target, mask)
            d, n = _reduce(fde, ade, fm)
            totals = ({k: totals.get(k, 0.0) + v * n for k, v in d.items()}
                      if totals else {k: v * n for k, v in d.items()})
            agents += n
    return {k: v / agents for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/mac/cross.npz")
    parser.add_argument("--checkpoint", default="data/mac/world_model_cross.pt")
    parser.add_argument("--scenario", default="cross",
                        choices=["cross", "merge", "roundabout"])
    parser.add_argument("--history_checkpoint", default="",
                        help="optional history-only WM for p(y|h) vs p(y|h,u) FDE")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 25, 100])
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--json", default=None, help="write metrics here for the paper build")
    parser.add_argument("--probes", default="",
                        help="comma-separated probe accelerations (default -4,0,2)")
    # The ground-truth channel must be the one that generated this dataset. Left
    # at the defaults it silently reports the v4 kernel for a v5 dataset.
    parser.add_argument("--beta_intent", type=float, default=BETA_INTENT)
    parser.add_argument("--beta_margin", type=float, default=BETA_MARGIN)
    parser.add_argument("--beta_bias", type=float, default=BETA_BIAS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    probe_accels = parse_probe_accels(args.probes, default=DEFAULT_PROBE_ACCELS)
    results = {}

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DiffusionWorldModel(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    data = np.load(args.dataset)
    val_loader = DataLoader(load_split(data, "val"), batch_size=args.batch_size)

    # ---------------------------------------------------------- displacement
    # Residual std over the training split calibrates the stochastic baseline so
    # it is a genuine generative competitor rather than a strawman.
    train_future = torch.from_numpy(data["future_train"]).float()
    train_hist = torch.from_numpy(data["history_train"]).float()
    tv = train_hist[:, -1, 1:, 2:4]
    horizon = train_future.shape[1]
    cv_train = tv[:, None, :, :] * (torch.arange(1, horizon + 1)[None, :, None, None] * 0.4)
    tmask = train_future[..., 2] > 0
    resid = (train_future[..., :2] - cv_train)[tmask]
    resid_std = float(resid.std())
    results["resid_std"] = resid_std
    results["n_train"] = int(data["history_train"].shape[0])
    results["n_val"] = int(data["history_val"].shape[0])
    if "episode_train" in data.files:
        results["n_train_episodes"] = int(np.unique(data["episode_train"]).size)
        results["n_val_episodes"] = int(np.unique(data["episode_val"]).size)
    tt = data["types_train"]
    results["labelled_frac"] = float((tt >= 0).mean())
    print(f"residual std over training split: {resid_std:.3f} m "
          f"(stochastic CV baseline uses this as its noise scale)\n")

    cv_det, cv_stoch, agents = {}, {}, 0
    det_sum, stoch_sum = None, None
    for i, batch in enumerate(val_loader):
        if i >= args.batches:
            break
        _, _, _, _, _, history_raw, future_raw = batch
        velocity = history_raw[:, -1, 1:, 2:4]
        cv = velocity[:, None, :, :] * (torch.arange(1, horizon + 1)[None, :, None, None] * 0.4)
        target = future_raw[..., :2].unsqueeze(1)
        mask = future_raw[..., 2].unsqueeze(1)

        det = cv.unsqueeze(1)
        fde, ade, fm = _fde_ade(det, target, mask)
        d, n = _reduce(fde, ade, fm)

        noise = torch.randn(cv.shape[0], args.n_samples, *cv.shape[1:]) * resid_std
        stoch = cv.unsqueeze(1) + noise
        fde, ade, fm = _fde_ade(stoch, target, mask)
        s, _ = _reduce(fde, ade, fm)

        det_sum = {k: det_sum.get(k, 0.0) + v * n for k, v in d.items()} if det_sum else {k: v * n for k, v in d.items()}
        stoch_sum = {k: stoch_sum.get(k, 0.0) + v * n for k, v in s.items()} if stoch_sum else {k: v * n for k, v in s.items()}
        agents += n

    cv_det = {k: v / agents for k, v in det_sum.items()}
    cv_stoch = {k: v / agents for k, v in stoch_sum.items()}
    results["cv_det"] = cv_det
    results["cv_stoch"] = cv_stoch
    results["diffusion"] = {}
    print(f"{'model':<28} {'minFDE':>8} {'meanFDE':>8} {'minADE':>8} {'meanADE':>8}")
    print(f"{'constant velocity (S=1)':<28} {cv_det['minFDE']:8.2f} {cv_det['meanFDE']:8.2f} "
          f"{cv_det['minADE']:8.2f} {cv_det['meanADE']:8.2f}")
    print(f"{f'CV + noise (S={args.n_samples})':<28} {cv_stoch['minFDE']:8.2f} {cv_stoch['meanFDE']:8.2f} "
          f"{cv_stoch['minADE']:8.2f} {cv_stoch['meanADE']:8.2f}")

    for steps in args.steps:
        m = _score_displacement(model, val_loader, device, args.n_samples, steps,
                                args.batches, zero_plan=False)
        results["diffusion"][str(steps)] = m
        print(f"{f'diffusion p(y|h,u) T={steps}':<28} {m['minFDE']:8.2f} {m['meanFDE']:8.2f} "
              f"{m['minADE']:8.2f} {m['meanADE']:8.2f}")

    # Same weights, plan zeroed: does u help predict the logged future?
    z = _score_displacement(model, val_loader, device, args.n_samples, 10,
                            args.batches, zero_plan=True)
    results["diffusion_zero_plan"] = z
    print(f"{'diffusion p(y|h) T=10':<28} {z['minFDE']:8.2f} {z['meanFDE']:8.2f} "
          f"{z['minADE']:8.2f} {z['meanADE']:8.2f}")

    if args.history_checkpoint:
        hckpt = torch.load(args.history_checkpoint, map_location=device, weights_only=False)
        hmodel = DiffusionWorldModel(**hckpt["config"]).to(device)
        hmodel.load_state_dict(hckpt["model"])
        hmodel.eval()
        h = _score_displacement(hmodel, val_loader, device, args.n_samples, 10,
                                args.batches, zero_plan=True)
        results["history"] = h
        print(f"{'history WM T=10':<28} {h['minFDE']:8.2f} {h['meanFDE']:8.2f} "
              f"{h['minADE']:8.2f} {h['meanADE']:8.2f}")

    # ------------------------------------------------------------- intention
    correct, total = 0, 0
    counts = torch.zeros(2)
    with torch.no_grad():
        for history, ego_plan, _, types, _, _, _ in val_loader:
            history, ego_plan, types = history.to(device), ego_plan.to(device), types.to(device)
            probs = model.predict_intentions(history, ego_plan)
            valid = types >= 0
            if valid.any():
                correct += int((probs.argmax(dim=-1)[valid] == types[valid]).sum())
                total += int(valid.sum())
                for c in (0, 1):
                    counts[c] += int((types[valid] == c).sum())
    majority = float(counts.max() / counts.sum()) if counts.sum() > 0 else 0.0
    results["intent_acc"] = correct / max(total, 1)
    results["intent_majority"] = majority
    results["intent_n"] = total
    print(f"\nintention accuracy: {correct / max(total, 1):.3f} "
          f"(majority class {majority:.3f}) over {total} labelled neighbours")

    # --------------------------------------------------------------- channel
    # The simulator's kernel makes a neighbour more likely to yield as the ego
    # accelerates. A model that has recovered the channel must show the same
    # monotone response when probed with constant-acceleration plans.
    print("\nchannel recovery (probe response of the learned model):")
    n_probe = min(20000, data["history_val"].shape[0])
    hist_raw = torch.from_numpy(data["history_val"][:n_probe]).float().to(device)
    types_probe = torch.from_numpy(data["types_val"][:n_probe]).long().to(device)
    hist = hist_raw.clone()
    hist[..., :2] /= checkpoint.get("pos_scale", POS_SCALE)
    hist[..., 2:4] /= checkpoint.get("vel_scale", VEL_SCALE)

    # The plan must start from the ego speed the history actually implies;
    # a fixed nominal speed makes the conditioning inconsistent with the scene.
    ego_speed = hist_raw[:, -1, 0, 2:4].norm(dim=-1).cpu().numpy()
    present = hist_raw[:, -1, 1:, 4] > 0
    deciding = types_probe >= 0
    print(f"  populations: {int(present.sum())} present neighbours, "
          f"{int(deciding.sum())} of them committing inside the plan horizon")

    # Trajectory shift is always measured vs the hold probe (a=0 when present),
    # not vs the first listed probe — otherwise assert-vs-hold is understated.
    hold_accel = 0.0 if 0.0 in [float(a) for a in probe_accels] else float(probe_accels[0])
    p_yields, p_yields_all = [], []
    sample_by_accel = {}
    with torch.no_grad():
        for accel in probe_accels:
            plans = np.stack([
                synthetic_plan(
                    float(s), accel, checkpoint["config"]["future_len"], 0.4,
                    SCENARIOS[args.scenario].max_speed)
                for s in ego_speed])
            plans[..., :2] /= checkpoint.get("pos_scale", POS_SCALE)
            plans[..., 2] /= checkpoint.get("vel_scale", VEL_SCALE)
            plan_t = torch.from_numpy(plans).float().to(device)

            probs = model.predict_intentions(
                hist, plan_t,
                guidance=float(checkpoint.get("guidance", 1.0)))[..., 0]
            p_all = float(probs[present].mean())
            p_dec = float(probs[deciding].mean()) if bool(deciding.any()) else float("nan")
            p_yields.append(p_dec)
            p_yields_all.append(p_all)

            torch.manual_seed(0)
            samples = decode_samples(
                model.sample(hist, plan_t, n_samples=8, steps=10), hist_raw)
            sample_by_accel[float(accel)] = samples
            if abs(float(accel) - hold_accel) < 1e-6:
                spread = float(samples.std(dim=1).mean())
        ref = sample_by_accel[hold_accel]
        shifts = []
        for accel in probe_accels:
            samples = sample_by_accel[float(accel)]
            if abs(float(accel) - hold_accel) < 1e-6:
                shift = 0.0
            else:
                shift = float((samples - ref).norm(dim=-1).mean())
            shifts.append(shift)
            print(f"  probe a={accel:+.1f} m/s^2  P(yield)={p_yields[len(shifts)-1]:.3f} (deciding) "
                  f"/ {p_yields_all[len(shifts)-1]:.3f} (all present)  "
                  f"trajectory shift vs a={hold_accel:+.1f}: {shift:.3f} m")
    print(f"  spread across diffusion samples: {spread:.3f} m "
          f"(a probe effect below this is not resolvable)")
    monotone = all(b >= a for a, b in zip(p_yields, p_yields[1:]))
    print(f"  P(yield) increases with ego acceleration: {monotone} "
          f"(ground-truth kernel says it must)")

    # Scene-conditional ground-truth channel. Older datasets lack the TTC margin
    # metadata and retain an explicitly labelled zero-margin fallback.
    if "priority_margin_val" in data.files:
        margins = torch.from_numpy(
            data["priority_margin_val"][:n_probe]).float()
        valid_truth = present.cpu()
        gt = []
        for accel in probe_accels:
            logits = (args.beta_margin * margins
                      + args.beta_intent * float(accel) + args.beta_bias)
            gt.append(float(torch.sigmoid(logits)[valid_truth].mean()))
        truth_kind = "scene-conditioned"
    else:
        kernel = SocialDriverManager(
            (0.0, 0.0), np.random.default_rng(0),
            beta_intent=args.beta_intent, beta_margin=args.beta_margin,
            beta_bias=args.beta_bias)
        gt = [kernel.yield_probability(0.0, a) for a in probe_accels]
        truth_kind = "zero-margin fallback"
    results["channel"] = {
        "probe_accels": list(probe_accels),
        "beta_intent": float(args.beta_intent),
        "beta_margin": float(args.beta_margin),
        "beta_bias": float(args.beta_bias),
        "p_yield_model": p_yields,
        "p_yield_model_all": p_yields_all,
        "p_yield_truth": gt,
        "truth_kind": truth_kind,
        "shifts": shifts,
        "spread": spread,
        "monotone": bool(monotone),
        "tv_truth": float(max(gt) - min(gt)),
        "tv_model": float(max(p_yields) - min(p_yields)),
        "tv_model_all": float(max(p_yields_all) - min(p_yields_all)),
    }
    print(f"  ground-truth kernel ({truth_kind}) over the same probes: "
          f"{', '.join(f'{p:.3f}' for p in gt)} (TV = {max(gt) - min(gt):.3f})")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
