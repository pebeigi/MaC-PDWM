"""Q2: interventional counterfactual prediction.

For a fixed scene h, execute several ego plans in the simulator (do(u)) and
compare world-model forecasts to the realised neighbour futures. A history-only
model must emit (almost) the same y for every u; a plan-conditioned model
should track the interventional shift.
"""
import argparse
import json
import os
import tempfile

import numpy as np
import torch

from mac.data.normalize import POS_SCALE, VEL_SCALE, decode_samples
from mac.data.scene import ego_yaw_rate, extract_future, extract_scene, synthetic_plan
from mac.envs.sumo_planning_env import EnvConfig, SumoPlanningEnv, parse_type_probs
from mac.models.belief import DEFAULT_PROBE_ACCELS, parse_probe_accels
from mac.models.diffusion_world_model import DiffusionWorldModel
from mac.policies.rule_based import TimeGapPolicy


def load_wm(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = DiffusionWorldModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def decision_relevant(env, radius):
    """True when the ego is on approach and a reactive driver is still undecided.

    Averaging the interventional shift over arbitrary states hides the channel:
    at most steps no neighbour can commit inside the horizon, so the true
    response is zero there by construction. The radius must sit outside the
    commitment distance, otherwise the only qualifying states are the ones where
    the driver has already decided and nothing is left to influence.
    """
    ego = env.egos[0]
    ego_state = env._last_snapshot.get(ego.veh_id)
    if ego_state is None or ego.done:
        return False
    if not 0.0 < env._signed_conflict_distance(ego_state) <= radius:
        return False
    point = np.asarray(env.spec.conflict_point)
    for veh_id, state in env._last_snapshot.items():
        if state["is_ego"] or env.drivers.types.get(veh_id) != "reactive":
            continue
        if veh_id in env.drivers.resolved:
            continue
        distance = float(np.linalg.norm(point - np.asarray([state["x"], state["y"]])))
        if distance <= radius:
            return True
    return False


def branch_reset(env, seed):
    rng = np.random.default_rng(int(seed))
    env.rng = rng
    env.drivers.rng = rng
    return env.reset(seed=int(seed))


def predict(model, history_raw, ego_speed, accels, device, n_samples, steps,
            max_speed, dt, zero_plan=False):
    hist = torch.from_numpy(history_raw).float().unsqueeze(0).to(device)
    hist_n = hist.clone()
    hist_n[..., :2] /= POS_SCALE
    hist_n[..., 2:4] /= VEL_SCALE
    plans = []
    yaw_rate = ego_yaw_rate(history_raw, dt)
    for a in accels:
        plan = synthetic_plan(
            ego_speed, a, model.future_len, dt, max_speed, yaw_rate=yaw_rate)
        plan[..., :2] /= POS_SCALE
        plan[..., 2] /= VEL_SCALE
        plans.append(plan)
    plan_t = torch.from_numpy(np.stack(plans)).float().to(device)
    if zero_plan:
        plan_t = torch.zeros_like(plan_t)
    hist_b = hist_n.expand(len(accels), -1, -1, -1)
    with torch.no_grad():
        samples = model.sample(
            hist_b, plan_t, n_samples=n_samples, steps=steps,
            eta=0.0, common_noise=True)
        pred = decode_samples(samples, hist.expand(len(accels), -1, -1, -1))
    return pred.cpu().numpy()  # (P, S, F, K, 2)


def fde_ade(pred, target, mask):
    # pred (S,F,K,2), target (F,K,2), mask (F,K).
    # Neighbours often leave before the last horizon step; score each agent at
    # its last valid frame instead of requiring mask[-1].
    err = np.linalg.norm(pred - target[None], axis=-1)  # (S,F,K)
    fde_parts, ade_parts = [], []
    for k in range(mask.shape[1]):
        steps = np.flatnonzero(mask[:, k] > 0)
        if steps.size == 0:
            continue
        last = int(steps[-1])
        fde_parts.append(err[:, last, k])
        ade_parts.append(err[:, steps, k].mean(axis=1))
    if not fde_parts:
        return np.nan, np.nan, np.nan, np.nan
    fde = np.stack(fde_parts, axis=1)
    ade = np.stack(ade_parts, axis=1)
    return float(np.min(fde)), float(np.mean(fde)), float(np.min(ade)), float(np.mean(ade))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_model", default="data/mac/world_model_cross.pt")
    parser.add_argument("--history_model",
                        default="data/mac/world_model_history_cross.pt")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--target_states", action="store_true",
                        help="branch only where a reactive driver can still "
                             "commit inside the queried horizon")
    parser.add_argument("--target_radius", type=float, default=55.0)
    parser.add_argument("--scenario", default="cross",
                        choices=["cross", "merge", "roundabout"])
    parser.add_argument("--bg_scale", type=float, default=1.5)
    parser.add_argument("--beta_intent", type=float, default=2.5)
    parser.add_argument("--beta_margin", type=float, default=0.3)
    parser.add_argument("--decision_distance", type=float, default=None)
    parser.add_argument("--intent_window", type=float, default=0.8)
    parser.add_argument("--type_probs", default="0.1,0.1,0.8")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--probes", default="-4,0,3",
                        help="action-aligned probes for do(u) roll-outs")
    parser.add_argument("--json", default="data/mac/wm_counterfactual.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    probes = parse_probe_accels(args.probes, default=DEFAULT_PROBE_ACCELS)
    device = torch.device(args.device)
    diff, dckpt = load_wm(args.world_model, device)
    hist_m = None
    try:
        hist_m, _ = load_wm(args.history_model, device)
    except FileNotFoundError:
        print(f"no history model at {args.history_model}")

    cfg = EnvConfig(
        scenario=args.scenario, seed=0, horizon=80,
        bg_rate_scale=args.bg_scale, beta_intent=args.beta_intent,
        beta_margin=args.beta_margin, decision_distance=args.decision_distance,
        intent_window=args.intent_window,
        type_probs=parse_type_probs(args.type_probs))
    env = SumoPlanningEnv(cfg, label="cf_eval", continuous=True)
    future_len = diff.future_len
    dt = env.dt
    max_speed = env.spec.max_speed

    rows = []
    try:
        for ep in range(args.episodes):
            seed = 10_000 + ep
            obs = branch_reset(env, seed)
            policy = TimeGapPolicy(env, noise=0.0)
            policy.reset()
            ok = True
            for step in range(args.warmup):
                idx = int(policy.act(obs)[0])
                accel = float(env.cfg.discrete_actions[idx])
                obs, _, dones, _ = env.step(np.array([accel]))
                if bool(np.all(dones)) or env.egos[0].done:
                    ok = False
                    break
                if args.target_states and step >= diff.history_len:
                    if decision_relevant(env, args.target_radius):
                        break
            if not ok or len(env.frames) < diff.history_len:
                continue
            if args.target_states and not decision_relevant(env, args.target_radius):
                continue
            ego_id = env.egos[0].veh_id
            scene = extract_scene(
                env.frames, len(env.frames) - 1, ego_id,
                diff.history_len, diff.n_neighbors,
                conflict_point=env.spec.conflict_point,
                approach_edge_prefixes=env.spec.approach_edge_prefixes,
                decision_distance=env.cfg.decision_distance)
            if scene is None or not scene["neighbor_ids"]:
                continue
            state = env._last_snapshot.get(ego_id)
            if state is None:
                continue
            ego_speed = float(state["speed"])
            history = scene["history"]
            neighbor_ids = scene["neighbor_ids"]

            pred_d = predict(diff, history, ego_speed, probes, device,
                             args.n_samples, args.sample_steps, max_speed, dt, False)
            pred_h = None
            if hist_m is not None:
                pred_h = predict(hist_m, history, ego_speed, probes, device,
                                 args.n_samples, args.sample_steps, max_speed, dt, True)

            # Branch from the saved simulator state rather than replaying a
            # fresh reset: vehicle ids embed an episode counter, so a replayed
            # episode contains different ids and every neighbour mask is empty.
            actual = {}
            decisions = {}
            with tempfile.TemporaryDirectory(prefix="mac_cf_") as tmp:
                state_path = os.path.join(tmp, "state.xml")
                branch_state = env.save_branch_state(state_path)
                t0 = len(env.frames) - 1
                for accel in probes:
                    env.load_branch_state(state_path, branch_state)
                    for _ in range(future_len):
                        _, _, dones, _ = env.step(np.array([float(accel)]))
                        if bool(np.all(dones)):
                            break
                    _, future = extract_future(
                        env.frames, t0, ego_id, neighbor_ids,
                        future_len, scene["origin"], scene["rot"],
                        diff.n_neighbors)
                    actual[float(accel)] = future
                    decisions[float(accel)] = [
                        env.drivers.resolved.get(v) for v in neighbor_ids]

            if len(actual) < 2:
                continue

            ref = probes[len(probes) // 2]
            gt_shift, d_shift, h_shift = [], [], []
            gt_peak, flips = [], []
            rec = {"episode": ep, "probes": {}}
            for i, accel in enumerate(probes):
                if float(accel) not in actual:
                    continue
                fut = actual[float(accel)]
                target, mask = fut[..., :2], fut[..., 2]
                md = fde_ade(pred_d[i], target, mask)
                rec["probes"][str(accel)] = {
                    "diffusion": dict(zip(("minFDE", "meanFDE", "minADE", "meanADE"), md)),
                }
                if pred_h is not None:
                    mh = fde_ade(pred_h[i], target, mask)
                    rec["probes"][str(accel)]["history"] = dict(
                        zip(("minFDE", "meanFDE", "minADE", "meanADE"), mh))
                if float(accel) != float(ref) and float(ref) in actual:
                    gt = actual[float(ref)]
                    w = (mask > 0) & (gt[..., 2] > 0)
                    if w.any():
                        gt_shift.append(float(np.linalg.norm(
                            target[w] - gt[..., :2][w], axis=-1).mean()))
                        # Averaging over every neighbour slot hides a large
                        # response on the one driver actually in conflict.
                        separation = np.linalg.norm(target - gt[..., :2], axis=-1)
                        gt_peak.append(float(np.where(w, separation, 0.0).max()))
                        ref_dec = decisions.get(float(ref), [])
                        cur_dec = decisions.get(float(accel), [])
                        flips.append(float(np.mean([
                            a != b for a, b in zip(ref_dec, cur_dec)
                        ])) if ref_dec and cur_dec else 0.0)
                        ref_i = list(probes).index(ref)
                        d_shift.append(float(np.linalg.norm(
                            pred_d[i].mean(axis=0)[w] - pred_d[ref_i].mean(axis=0)[w],
                            axis=-1).mean()))
                        if pred_h is not None:
                            h_shift.append(float(np.linalg.norm(
                                pred_h[i].mean(axis=0)[w] - pred_h[ref_i].mean(axis=0)[w],
                                axis=-1).mean()))
            rec["gt_shift"] = float(np.mean(gt_shift)) if gt_shift else None
            rec["gt_peak_shift"] = float(np.mean(gt_peak)) if gt_peak else None
            rec["decision_flip_rate"] = float(np.mean(flips)) if flips else None
            rec["diffusion_shift"] = float(np.mean(d_shift)) if d_shift else None
            rec["history_shift"] = float(np.mean(h_shift)) if h_shift else None
            rows.append(rec)
            print(f"ep {ep:3d}  n={len(rec['probes'])}  GTΔ={rec['gt_shift']}  "
                  f"diffΔ={rec['diffusion_shift']}  histΔ={rec['history_shift']}",
                  flush=True)
    finally:
        env.close()

    def _finite_mean(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    def _mean_metric(model_key, field):
        vals = []
        for rec in rows:
            for p in rec["probes"].values():
                if model_key in p and np.isfinite(p[model_key][field]):
                    vals.append(p[model_key][field])
        return float(np.mean(vals)) if vals else None

    summary = {
        "n_scenes": len(rows),
        "probes": list(probes),
        "gt_shift": _finite_mean([r["gt_shift"] for r in rows]),
        "gt_peak_shift": _finite_mean([r.get("gt_peak_shift") for r in rows]),
        "decision_flip_rate": _finite_mean(
            [r.get("decision_flip_rate") for r in rows]),
        "diffusion_shift": _finite_mean([r["diffusion_shift"] for r in rows]),
        "history_shift": _finite_mean([r["history_shift"] for r in rows]),
        "diffusion_meanFDE": _mean_metric("diffusion", "meanFDE"),
        "diffusion_minFDE": _mean_metric("diffusion", "minFDE"),
        "history_meanFDE": _mean_metric("history", "meanFDE"),
        "history_minFDE": _mean_metric("history", "minFDE"),
    }
    blob = {"summary": summary, "episodes": rows}
    with open(args.json, "w") as handle:
        json.dump(blob, handle, indent=2)
    print("\nsummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
