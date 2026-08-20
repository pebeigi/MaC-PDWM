"""Collect interaction episodes for training the world model.

The world model is queried counterfactually at planning time, so the behaviour
policy used here has to make the ego's plan vary for a *fixed* scene. A purely
reactive controller makes the plan a deterministic function of the history, and
the learned conditional would then be observational rather than interventional.
Exogenous variation comes from four sources:

* ~8% constant-speed episodes,
* ~20% constant-acceleration episodes covering the discrete action set (and
  the planner probes $\{-4,0,+2\}$),
* ~20% piecewise open-loop $F$-step plans (identifies $\mathrm{do}(u_{1:F})$),
* ~52% gap-acceptance episodes with per-action $\epsilon\sim\mathcal{U}(0.15,0.45)$.
"""
import argparse
import os
import pickle
import time

import numpy as np

from mac.envs.sumo_planning_env import EnvConfig, SumoPlanningEnv, parse_type_probs
from mac.models.belief import DEFAULT_PROBE_ACCELS
from mac.policies.rule_based import (ConstantAccelPolicy, ConstantSpeedPolicy,
                                     OpenLoopPlanPolicy, TimeGapPolicy)

F_PLAN = 20  # matches world-model / probe horizon


def make_policy(env, rng):
    """Sample a behaviour for one episode."""
    roll = rng.random()
    actions = [float(a) for a in env.cfg.discrete_actions]
    # Probes are a subset of the discrete set; keep the union explicit.
    choices = sorted(set(actions) | set(float(a) for a in DEFAULT_PROBE_ACCELS))
    if roll < 0.08:
        return ConstantSpeedPolicy(env), "constant"
    if roll < 0.28:
        accel = float(choices[int(rng.integers(0, len(choices)))])
        return ConstantAccelPolicy(env, accel), f"hold(a={accel:+.1f})"
    if roll < 0.48:
        # Piecewise open-loop $F$-step plans: identifies do(u_{1:F}), not just
        # one-step p(y | h, u_t).
        n_seg = int(rng.integers(2, 5))
        seq = []
        for _ in range(n_seg):
            a = float(choices[int(rng.integers(0, len(choices)))])
            seq.extend([a] * F_PLAN)
        return OpenLoopPlanPolicy(env, seq), f"plan(seg={n_seg})"
    aggressiveness = float(rng.uniform(-0.4, 1.0))
    accept_gap = float(rng.uniform(1.0, 2.5))
    # Randomisation is deliberately strong: it is the intervention that makes
    # p(y | h, u) identifiable from observational roll-outs.
    noise = float(rng.uniform(0.15, 0.45))
    policy = TimeGapPolicy(env, accept_gap=accept_gap, aggressiveness=aggressiveness,
                           noise=noise, rng=rng)
    return policy, f"gap(a={aggressiveness:.2f},g={accept_gap:.2f},n={noise:.2f})"


def collect(out_dir, episodes, scenario, seed, bg_scale_range=(0.7, 2.5),
            beta_intent=2.5, type_probs=(0.1, 0.1, 0.8), beta_margin=0.3,
            decision_distance=None, intent_window=0.8):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    cfg = EnvConfig(scenario=scenario, seed=seed, horizon=150,
                    beta_intent=beta_intent, type_probs=type_probs,
                    beta_margin=beta_margin, decision_distance=decision_distance,
                    intent_window=intent_window)
    env = SumoPlanningEnv(cfg, label=f"collect_{scenario}_{seed}")

    manifest = []
    started = time.time()
    try:
        for ep in range(episodes):
            env.cfg.bg_rate_scale = float(rng.uniform(*bg_scale_range))
            obs = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            policy, tag = make_policy(env, rng)
            if hasattr(policy, "reset"):
                policy.reset()

            total = 0.0
            for _ in range(cfg.horizon):
                obs, rewards, dones, info = env.step(policy.act(obs))
                total += float(np.sum(rewards))
                if bool(np.all(dones)):
                    break

            record = {
                "frames": env.frames,
                "ego_ids": [ego.veh_id for ego in env.egos],
                "outcome": info["outcomes"][0] or "timeout",
                "driver_types": dict(env.drivers.types),
                "resolved": dict(env.drivers.resolved),
                "resolved_step": dict(env.drivers.resolved_step),
                "policy": tag,
                "bg_rate_scale": env.cfg.bg_rate_scale,
                "dt": env.dt,
                "scenario": scenario,
                "return": total,
            }
            path = os.path.join(out_dir, f"ep_{seed:04d}_{ep:05d}.pkl")
            with open(path, "wb") as handle:
                pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
            manifest.append({"path": path, "outcome": record["outcome"], "policy": tag})

            if (ep + 1) % 25 == 0:
                elapsed = time.time() - started
                print(f"  {ep + 1}/{episodes} episodes ({elapsed:.0f}s, "
                      f"{elapsed / (ep + 1):.2f}s/ep)", flush=True)
    finally:
        env.close()

    outcomes = {}
    for item in manifest:
        outcomes[item["outcome"]] = outcomes.get(item["outcome"], 0) + 1
    print(f"collected {len(manifest)} episodes into {out_dir}: {outcomes}")
    return manifest


def collect_paired(out_dir, episodes, scenario, seed,
                   bg_scale_range=(0.7, 2.5), beta_intent=2.5,
                   type_probs=(0.1, 0.1, 0.8), beta_margin=0.3,
                   decision_distance=None, intent_window=0.8,
                   probes=DEFAULT_PROBE_ACCELS):
    """Collect matched do(u) branches from replay-identical histories."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    cfg = EnvConfig(
        scenario=scenario, seed=seed, horizon=150,
        beta_intent=beta_intent, type_probs=type_probs,
        beta_margin=beta_margin, decision_distance=decision_distance,
        intent_window=intent_window)
    env = SumoPlanningEnv(cfg, label=f"paired_{scenario}_{seed}")
    commit_distance = float(env.cfg.decision_distance)
    written = 0
    try:
        for ep in range(episodes):
            episode_seed = int(rng.integers(0, 2**31 - 1))
            bg_scale = float(rng.uniform(*bg_scale_range))
            env.cfg.bg_rate_scale = bg_scale
            obs = env.reset(seed=episode_seed)
            policy = TimeGapPolicy(
                env, accept_gap=float(rng.uniform(1.0, 2.5)),
                aggressiveness=float(rng.uniform(-0.4, 1.0)),
                noise=0.0, rng=rng)
            policy.reset()
            prefix = []
            valid = True
            found_decision_scene = False
            for step in range(80):
                action = int(policy.act(obs)[0])
                prefix.append(action)
                obs, _, dones, _ = env.step(np.asarray([action]))
                if bool(dones[0]):
                    valid = False
                    break
                ego = env.egos[0]
                ego_state = env._last_snapshot.get(ego.veh_id)
                if step < 4 or ego_state is None:
                    continue
                ego_distance = env._signed_conflict_distance(ego_state)
                # The search radius has to sit outside the commitment distance.
                # A driver resolves the moment it enters the decision zone, so
                # looking for one that is undecided *inside* that zone finds
                # nothing and no branches are ever collected.
                search_radius = commit_distance + 20.0
                reactive_near = any(
                    env.drivers.types.get(veh_id) == "reactive"
                    and veh_id not in env.drivers.resolved
                    and env.drivers._edge_allowed(state)
                    and float(np.linalg.norm(
                        np.asarray(env.spec.conflict_point)
                        - np.asarray([state["x"], state["y"]]))) <= search_radius
                    for veh_id, state in env._last_snapshot.items()
                    if not state["is_ego"])
                if 0.0 < ego_distance <= search_radius and reactive_near:
                    found_decision_scene = True
                    break
            if not valid or not found_decision_scene:
                continue
            sample_t = len(env.frames) - 1

            for probe in probes:
                env.cfg.bg_rate_scale = bg_scale
                obs = env.reset(seed=episode_seed)
                dead = False
                for action in prefix:
                    obs, _, dones, _ = env.step(np.asarray([action]))
                    if bool(dones[0]):
                        dead = True
                        break
                if dead:
                    continue
                probe_action = int(np.argmin(np.abs(
                    np.asarray(env.cfg.discrete_actions) - float(probe))))
                for _ in range(F_PLAN):
                    obs, _, dones, _ = env.step(np.asarray([probe_action]))
                    if bool(dones[0]):
                        break
                record = {
                    "frames": env.frames,
                    "ego_ids": [ego.veh_id for ego in env.egos],
                    "outcome": env.egos[0].outcome or "truncated",
                    "driver_types": dict(env.drivers.types),
                    "resolved": dict(env.drivers.resolved),
                    "resolved_step": dict(env.drivers.resolved_step),
                    "policy": f"paired(a={float(probe):+.1f})",
                    "interventional": True,
                    "sample_times": [sample_t],
                    "intervention_id": f"{seed}:{ep}",
                    "probe_accel": float(probe),
                    "bg_rate_scale": bg_scale,
                    "dt": env.dt,
                    "scenario": scenario,
                    "channel": {
                        "beta_intent": beta_intent,
                        "beta_margin": beta_margin,
                        "decision_distance": commit_distance,
                        "intent_window": intent_window,
                        "type_probs": tuple(type_probs),
                        "approach_edge_prefixes": tuple(
                            env.spec.approach_edge_prefixes),
                    },
                }
                path = os.path.join(
                    out_dir,
                    f"pair_{seed:04d}_{ep:05d}_{float(probe):+04.1f}.pkl")
                with open(path, "wb") as handle:
                    pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
                written += 1
    finally:
        env.close()
    print(f"collected {written} matched branches into {out_dir}")
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/mac/raw/cross")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--paired", action="store_true",
                        help="collect matched probe branches from identical histories")
    parser.add_argument("--scenario", default="cross",
                        choices=["cross", "merge", "roundabout"])
    parser.add_argument("--seed", type=int, default=0)
    # Collecting with 0.0 produces the matched dataset for the severed-channel
    # control, so its world model is not trained on a channel it will never see.
    parser.add_argument("--beta_intent", type=float, default=2.5)
    parser.add_argument("--beta_margin", type=float, default=0.3)
    parser.add_argument("--decision_distance", type=float, default=None,
                        help="commit radius; default = scenario-specific "
                             "(cross/merge 35, roundabout 22)")
    parser.add_argument("--intent_window", type=float, default=0.8,
                        help="EMA window of the ego-motion intent signal (s)")
    parser.add_argument("--type_probs", default="0.1,0.1,0.8",
                        help="yielder,contester,reactive probabilities")
    args = parser.parse_args()

    collector = collect_paired if args.paired else collect
    collector(args.out_dir, args.episodes, args.scenario, args.seed,
              beta_intent=args.beta_intent, beta_margin=args.beta_margin,
              decision_distance=args.decision_distance,
              intent_window=args.intent_window,
              type_probs=parse_type_probs(args.type_probs))


if __name__ == "__main__":
    main()
