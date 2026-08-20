"""How far ahead must a probe reach before the ego's plan changes anything?

The belief encoder queries the world model with short constant-acceleration
plans. If a neighbour's commitment is decided over a longer span than the probe
covers, those queries are answered almost identically for every candidate plan
and the learned belief cannot carry information the geometry baseline lacks.

This measures, from matched branches of identical scenes, how the decision-flip
rate and the trajectory separation between an assertive and a yielding probe
grow with the probe horizon.
"""
import argparse
import json
import os
import tempfile

import numpy as np

from mac.envs.sumo_planning_env import EnvConfig, SumoPlanningEnv
from mac.policies.rule_based import TimeGapPolicy


def decision_relevant(env, radius):
    ego = env.egos[0]
    state = env._last_snapshot.get(ego.veh_id)
    if state is None or ego.done:
        return False
    if not 0.0 < env._signed_conflict_distance(state) <= radius:
        return False
    point = np.asarray(env.spec.conflict_point)
    for veh_id, other in env._last_snapshot.items():
        if other["is_ego"] or env.drivers.types.get(veh_id) != "reactive":
            continue
        if veh_id in env.drivers.resolved:
            continue
        distance = float(np.linalg.norm(point - np.array([other["x"], other["y"]])))
        if distance <= radius:
            return True
    return False


def roll_branch(env, path, state, accel, checkpoints, tracked):
    """Roll one probe, recording decisions and positions at each checkpoint."""
    env.load_branch_state(path, state)
    trace = {}
    for step in range(1, max(checkpoints) + 1):
        _, _, dones, _ = env.step(np.array([float(accel)], dtype=np.float32))
        if step in checkpoints:
            trace[step] = {
                veh_id: (
                    env.drivers.resolved.get(veh_id),
                    (env._last_snapshot[veh_id]["x"],
                     env._last_snapshot[veh_id]["y"])
                    if veh_id in env._last_snapshot else None)
                for veh_id in tracked
            }
        if bool(np.all(dones)):
            for later in checkpoints:
                trace.setdefault(later, trace.get(step, {}))
            break
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="cross,merge,roundabout")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--radius", type=float, default=55.0)
    ap.add_argument("--assert_accel", type=float, default=3.0)
    ap.add_argument("--yield_accel", type=float, default=-4.0)
    ap.add_argument("--checkpoints", default="5,10,15,20,25,30,40")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    checkpoints = [int(c) for c in args.checkpoints.split(",")]
    results = {}
    for scenario in args.scenarios.split(","):
        cfg = EnvConfig(scenario=scenario)
        env = SumoPlanningEnv(cfg, label=f"horizon_{scenario}", continuous=True)
        dt = env.dt
        flips = {c: [] for c in checkpoints}
        seps = {c: [] for c in checkpoints}
        scenes = 0
        try:
            for ep in range(args.episodes):
                obs = env.reset(seed=500_000 + ep)
                policy = TimeGapPolicy(env, noise=0.0)
                policy.reset()
                ready = False
                for step in range(args.warmup):
                    idx = int(policy.act(obs)[0])
                    obs, _, dones, _ = env.step(
                        np.array([float(env.cfg.discrete_actions[idx])]))
                    if bool(np.all(dones)) or env.egos[0].done:
                        break
                    if step > 5 and decision_relevant(env, args.radius):
                        ready = True
                        break
                if not ready:
                    continue
                tracked = [
                    veh_id for veh_id, other in env._last_snapshot.items()
                    if not other["is_ego"]
                    and env.drivers.types.get(veh_id) == "reactive"
                    and veh_id not in env.drivers.resolved]
                if not tracked:
                    continue
                scenes += 1
                with tempfile.TemporaryDirectory(prefix="mac_hz_") as tmp:
                    path = os.path.join(tmp, "state.xml")
                    state = env.save_branch_state(path)
                    a = roll_branch(env, path, state, args.assert_accel,
                                    checkpoints, tracked)
                    b = roll_branch(env, path, state, args.yield_accel,
                                    checkpoints, tracked)
                for c in checkpoints:
                    ta, tb = a.get(c, {}), b.get(c, {})
                    flipped, separations = [], []
                    for veh_id in tracked:
                        da, pa = ta.get(veh_id, (None, None))
                        db, pb = tb.get(veh_id, (None, None))
                        flipped.append(da != db)
                        if pa is not None and pb is not None:
                            separations.append(float(np.linalg.norm(
                                np.array(pa) - np.array(pb))))
                    if flipped:
                        flips[c].append(float(np.mean(flipped)))
                    if separations:
                        seps[c].append(float(np.max(separations)))
        finally:
            env.close()

        rows = []
        for c in checkpoints:
            rows.append({
                "steps": c,
                "seconds": round(c * dt, 1),
                "flip_rate": float(np.mean(flips[c])) if flips[c] else None,
                "peak_separation": float(np.mean(seps[c])) if seps[c] else None,
            })
        results[scenario] = {"scenes": scenes, "horizons": rows}

        print(f"\n=== {scenario} ({scenes} scenes) ===")
        print(f"{'horizon':>9}{'flip rate':>12}{'peak sep (m)':>15}")
        for row in rows:
            flip = "-" if row["flip_rate"] is None else f"{row['flip_rate']:.3f}"
            sep = "-" if row["peak_separation"] is None else f"{row['peak_separation']:.2f}"
            print(f"{row['seconds']:>7.1f}s{flip:>12}{sep:>15}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
