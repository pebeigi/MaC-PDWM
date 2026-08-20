"""Closed-loop rule-based baselines, reported in the same format as PPO.

Uses the gap-acceptance and always-go controllers already used for world-model
data collection, so the comparison is against behaviours the WM actually saw.
"""
import argparse
import collections
import json
import os

import numpy as np

from mac.envs.sumo_planning_env import EnvConfig, SumoPlanningEnv, parse_type_probs
from mac.policies.rule_based import ConstantSpeedPolicy, TimeGapPolicy


def evaluate(env, policy, episodes, seed_offset=100000):
    outcomes = collections.Counter()
    returns, steps, brakes, success_steps = [], [], [], []
    for ep in range(episodes):
        obs = env.reset(seed=seed_offset + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        total = 0.0
        for _ in range(env.cfg.horizon):
            obs, rewards, dones, info = env.step(policy.act(obs))
            total += float(np.sum(rewards))
            if bool(np.all(dones)):
                break
        outcome = info["outcomes"][0] or "timeout"
        outcomes[outcome] += 1
        returns.append(total)
        steps.append(info["step"])
        brakes.append(info["driver_stats"]["ego_induced_brakes"])
        if outcome == "success":
            success_steps.append(info["step"])
    return {
        "return": float(np.mean(returns)),
        "steps": float(np.mean(steps)),
        "crossing_steps": float(np.mean(success_steps)) if success_steps else float("nan"),
        "induced_brakes": float(np.mean(brakes)),
        "success_rate": outcomes["success"] / episodes,
        "collision_rate": outcomes["collision"] / episodes,
        "timeout_rate": outcomes["timeout"] / episodes,
        "lost_rate": outcomes["lost"] / episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="cross",
                        choices=["cross", "merge", "roundabout"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--bg_scale", type=float, default=1.5)
    parser.add_argument("--beta_intent", type=float, default=0.45)
    parser.add_argument("--type_probs", default="0.4,0.25,0.35")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="data/mac/baselines")
    args = parser.parse_args()

    cfg = EnvConfig(scenario=args.scenario, seed=args.seed, horizon=150,
                    bg_rate_scale=args.bg_scale, beta_intent=args.beta_intent,
                    type_probs=parse_type_probs(args.type_probs))
    env = SumoPlanningEnv(cfg, label=f"baseline_{args.scenario}_{args.seed}")
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        for name, factory in (
            ("gap", lambda: TimeGapPolicy(env)),
            ("constant", lambda: ConstantSpeedPolicy(env)),
        ):
            policy = factory()
            metrics = evaluate(env, policy, args.episodes)
            blob = {
                "config": {**vars(args), "belief": name},
                "history": [{**metrics, "iteration": 0}],
            }
            path = os.path.join(args.out_dir, f"metrics_{name}_seed{args.seed}.json")
            with open(path, "w") as handle:
                json.dump(blob, handle, indent=2)
            print(f"[{name}] {json.dumps(metrics)}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
