"""Quick end-to-end check that the planning env runs and the baselines differ."""
import argparse
import collections

import numpy as np

from mac.envs.sumo_planning_env import EnvConfig, SumoPlanningEnv
from mac.policies.rule_based import ConstantSpeedPolicy, TimeGapPolicy


def run(policy_name, scenario, episodes, seed, bg_scale=1.0):
    cfg = EnvConfig(scenario=scenario, seed=seed, horizon=120, bg_rate_scale=bg_scale)
    env = SumoPlanningEnv(cfg, label=f"smoke_{policy_name}_{scenario}")
    policy = TimeGapPolicy(env) if policy_name == "gap" else ConstantSpeedPolicy(env)

    outcomes = collections.Counter()
    returns, steps_to_finish, brakes = [], [], []
    try:
        for ep in range(episodes):
            obs = env.reset(seed=seed + ep)
            if hasattr(policy, "reset"):
                policy.reset()
            total = 0.0
            for _ in range(cfg.horizon):
                obs, rewards, dones, info = env.step(policy.act(obs))
                total += float(np.sum(rewards))
                if bool(np.all(dones)):
                    break
            outcomes[info["outcomes"][0] or "timeout"] += 1
            returns.append(total)
            steps_to_finish.append(info["step"])
            brakes.append(info["driver_stats"]["ego_induced_brakes"])
    finally:
        env.close()

    print(f"[{scenario}/{policy_name}] bg_scale={bg_scale} episodes={episodes} "
          f"return={np.mean(returns):6.2f} steps={np.mean(steps_to_finish):5.1f} "
          f"induced_hard_brakes={np.mean(brakes):4.1f} outcomes={dict(outcomes)}")
    return outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="cross",
                        choices=["cross", "merge", "roundabout"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bg_scale", type=float, nargs="+", default=[1.0])
    args = parser.parse_args()

    for bg_scale in args.bg_scale:
        for policy_name in ("gap", "constant"):
            run(policy_name, args.scenario, args.episodes, args.seed, bg_scale)


if __name__ == "__main__":
    main()
