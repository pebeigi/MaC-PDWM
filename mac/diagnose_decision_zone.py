"""Short SUMO smoke check for decision-zone / neighbor ranking on cross & RA.

Reports commit-edge mix, ego–other distance at commit, and whether the K=5
slots include approaching conflict partners. Run before long collection.

    python -m mac.diagnose_decision_zone --scenarios cross,roundabout --episodes 20
"""
import argparse
import collections

import numpy as np

from mac.envs.sumo_planning_env import SCENARIOS, EnvConfig, SumoPlanningEnv
from mac.policies.rule_based import TimeGapPolicy


def edge_bucket(edge):
    if not edge:
        return "?"
    if edge.startswith(":"):
        parts = edge.split("_")
        return parts[0] + "_*" if parts else edge
    return edge


def run_scenario(scenario, episodes, seed, bg_scale):
    spec = SCENARIOS[scenario]
    cfg = EnvConfig(scenario=scenario, seed=seed, horizon=120,
                    bg_rate_scale=bg_scale)
    env = SumoPlanningEnv(cfg, label=f"dz_{scenario}")
    edge_counts = collections.Counter()
    ego_other_dist = []
    k5_approaching = []
    k5_edges = collections.Counter()
    commits = 0
    labelled_slots = 0
    total_slots = 0
    reactive_commits = 0
    try:
        for ep in range(episodes):
            obs = env.reset(seed=seed + ep)
            policy = TimeGapPolicy(
                env, accept_gap=1.6, aggressiveness=0.3,
                noise=0.1, rng=np.random.default_rng(seed + ep))
            policy.reset()
            seen = set()
            for _ in range(cfg.horizon):
                obs, _, dones, _ = env.step(policy.act(obs))
                ego = env.egos[0]
                ego_state = env._last_snapshot.get(ego.veh_id)
                for veh_id in list(env.drivers.resolved):
                    if veh_id in seen:
                        continue
                    seen.add(veh_id)
                    commits += 1
                    if env.drivers.types.get(veh_id) == "reactive":
                        reactive_commits += 1
                    state = env._last_snapshot.get(veh_id)
                    if state is None or ego_state is None:
                        continue
                    edge_counts[edge_bucket(str(state.get("edge", "?")))] += 1
                    ego_other_dist.append(float(np.hypot(
                        state["x"] - ego_state["x"],
                        state["y"] - ego_state["y"])))
                if ego_state is not None:
                    d_ego = env._signed_conflict_distance(ego_state)
                    if 0.0 < d_ego <= env.cfg.decision_distance + 15.0:
                        slot_ids = env._neighbor_ids(
                            ego_state, env._last_snapshot, ego.veh_id)
                        for vid in slot_ids:
                            other = env._last_snapshot.get(vid)
                            if other is not None:
                                k5_edges[edge_bucket(str(other.get("edge", "?")))] += 1
                        feats = env._neighbor_features(
                            ego_state, env._last_snapshot, ego.veh_id)
                        n_dim = env.neighbor_feature_dim
                        approaching = 0
                        for k in range(env.cfg.n_neighbors):
                            total_slots += 1
                            slot = feats[k * n_dim:(k + 1) * n_dim]
                            if slot[-1] <= 0:
                                continue
                            labelled_slots += 1
                            if slot[5] > 0:
                                approaching += 1
                        k5_approaching.append(approaching)
                if bool(np.all(dones)):
                    break
    finally:
        env.close()

    print(f"\n=== {scenario} ===")
    print(f"  conflict={spec.conflict_point} D={spec.decision_distance}")
    print(f"  approach_edge_prefixes={spec.approach_edge_prefixes or '(any)'}")
    print(f"  commits={commits} reactive={reactive_commits} over {episodes} eps")
    if ego_other_dist:
        arr = np.asarray(ego_other_dist)
        print(f"  ego–other dist at commit: median={np.median(arr):.1f} "
              f"p90={np.percentile(arr, 90):.1f}")
    print(f"  commit edges: {dict(edge_counts.most_common(12))}")
    print(f"  K=5 slot edges: {dict(k5_edges.most_common(12))}")
    if k5_approaching:
        arr = np.asarray(k5_approaching, dtype=float)
        print(f"  K=5 approaching partners (mean/median): "
              f"{arr.mean():.2f}/{np.median(arr):.1f}")
    if total_slots:
        print(f"  occupied neighbor slots: {labelled_slots}/{total_slots} "
              f"({100.0 * labelled_slots / total_slots:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="cross,roundabout")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bg_scale", type=float, default=1.5)
    args = ap.parse_args()
    for scenario in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
        run_scenario(scenario, args.episodes, args.seed, args.bg_scale)


if __name__ == "__main__":
    main()
