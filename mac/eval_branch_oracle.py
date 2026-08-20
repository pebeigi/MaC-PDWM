"""Exact matched-branch upper bound for the negotiation task.

At each decision point, SUMO and all Python-side environment state are saved.
Every candidate constant-acceleration plan is then executed from that identical
state with identical random numbers.  The best realised branch is committed.
This is intentionally non-causal and expensive: it is a gate that establishes
whether the environment contains action-dependent value before a learned world
model is trained to approximate it.
"""
import argparse
import collections
import json
import os
import tempfile

import numpy as np

from mac.envs.sumo_planning_env import EnvConfig, SumoPlanningEnv, parse_type_probs
from mac.models.belief import DEFAULT_PROBE_ACCELS, parse_probe_accels
from mac.policies.rule_based import TimeGapPolicy


def continuation_value(env, exit_distance):
    """Value of the unrolled remainder, assuming the ego holds its current speed.

    A distance-shaped stand-in makes stalling look cheap: creeping short of the
    conflict point avoids the proximity and collision terms while the shaping
    term barely moves, so the greedy branch policy times out and scores below a
    plain rule-based baseline instead of bounding it.
    """
    ego = env.egos[0]
    state = env._last_snapshot.get(ego.veh_id)
    if state is None:
        return 0.0
    remaining_steps = max(env.cfg.horizon - env.step_count, 0)
    distance = max(env._signed_conflict_distance(state), 0.0) + exit_distance
    eta_steps = distance / max(float(state["speed"]), 0.5) / env.dt
    if eta_steps <= remaining_steps:
        return env.cfg.success_reward - env.cfg.time_penalty * eta_steps
    return -env.cfg.timeout_penalty - env.cfg.time_penalty * remaining_steps


def branch_score(env, accel, horizon, exit_distance, rollout_steps=0,
                 base_committed=None):
    """Score a candidate by committing to it, then rolling out the base policy.

    A one-window lookahead closed with an analytic tail is not a bound: the tail
    is flat once it projects a timeout, so braking costs nothing and the search
    stalls short of the conflict. Completing each branch with the rule-based
    policy makes this rollout policy improvement, which cannot score below that
    policy except through sampling noise.

    The base policy latches its gap acceptance, so its latch is part of the
    branch state: starting each tail from a cleared latch makes an ego that is
    already committed re-judge the crossing from inside the junction, brake, and
    drag every branch down to the same bad outcome.
    """
    total = 0.0
    done = False
    info = {}
    obs = None
    for _ in range(horizon):
        obs, rewards, dones, info = env.step(np.asarray([accel], dtype=np.float32))
        total += float(rewards[0])
        done = bool(dones[0])
        if done:
            break
    if not done and rollout_steps > 0:
        policy = TimeGapPolicy(env, noise=0.0)
        policy.committed = dict(base_committed or {})
        for _ in range(rollout_steps):
            idx = int(policy.act(obs)[0])
            base = float(env.cfg.discrete_actions[idx])
            obs, rewards, dones, info = env.step(np.asarray([base], dtype=np.float32))
            total += float(rewards[0])
            done = bool(dones[0])
            if done:
                break
    if not done:
        total += continuation_value(env, exit_distance)
    return total, done, info


def run_branch_oracle(env, seed, probes, commit_steps, exit_distance=30.0,
                      rollout_steps=0):
    obs = env.reset(seed=seed)
    total = 0.0
    done = False
    info = {}
    tracker = TimeGapPolicy(env, noise=0.0)
    tracker.reset()
    with tempfile.TemporaryDirectory(prefix="mac_branch_") as tmp:
        while not done:
            path = os.path.join(tmp, "state.xml")
            state = env.save_branch_state(path)
            committed = dict(tracker.committed)
            scores = []
            for accel in probes:
                env.load_branch_state(path, state)
                score, _, _ = branch_score(env, accel, commit_steps,
                                           exit_distance, rollout_steps,
                                           committed)
                scores.append(score)
            chosen = float(probes[int(np.argmax(scores))])
            env.load_branch_state(path, state)
            for _ in range(commit_steps):
                # Keep the base policy's latch on the executed trajectory so the
                # tails scored at the next decision start from the right state.
                tracker.act(obs)
                obs, rewards, dones, info = env.step(
                    np.asarray([chosen], dtype=np.float32))
                total += float(rewards[0])
                done = bool(dones[0])
                if done:
                    break
    return total, info


def run_time_gap(env, seed):
    obs = env.reset(seed=seed)
    policy = TimeGapPolicy(env, noise=0.0)
    policy.reset()
    total = 0.0
    info = {}
    for _ in range(env.cfg.horizon):
        idx = int(policy.act(obs)[0])
        accel = float(env.cfg.discrete_actions[idx])
        obs, rewards, dones, info = env.step(
            np.asarray([accel], dtype=np.float32))
        total += float(rewards[0])
        if bool(dones[0]):
            break
    return total, info


def summarize(rows):
    outcomes = collections.Counter(row["outcome"] for row in rows)
    n = max(len(rows), 1)
    return {
        "episodes": len(rows),
        "return": float(np.mean([row["return"] for row in rows])) if rows else None,
        "success_rate": outcomes["success"] / n,
        "collision_rate": outcomes["collision"] / n,
        "timeout_rate": outcomes["timeout"] / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="cross",
                    choices=["cross", "merge", "roundabout"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--commit_steps", type=int, default=10)
    ap.add_argument("--probes", default="all",
                    help="'all' searches the planner's own action set, which is "
                         "the bound the learned arms are actually competing with")
    ap.add_argument("--rollout_steps", type=int, default=40,
                    help="steps of base-policy rollout appended to each branch")
    ap.add_argument("--exit_distance", type=float, default=30.0,
                    help="route length past the conflict point, used by the "
                         "continuation value to price a timeout")
    ap.add_argument("--bg_scale", type=float, default=1.5)
    ap.add_argument("--beta_intent", type=float, default=2.5)
    ap.add_argument("--beta_margin", type=float, default=0.3)
    ap.add_argument("--decision_distance", type=float, default=None)
    ap.add_argument("--intent_window", type=float, default=0.8)
    ap.add_argument("--type_probs", default="0.1,0.1,0.8")
    ap.add_argument("--courtesy_grace_steps", type=int, default=3)
    ap.add_argument("--time_penalty", type=float, default=0.06)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if args.probes.strip().lower() == "all":
        probes = None
    else:
        probes = parse_probe_accels(args.probes, default=DEFAULT_PROBE_ACCELS)
    cfg = EnvConfig(
        scenario=args.scenario, bg_rate_scale=args.bg_scale,
        beta_intent=args.beta_intent, beta_margin=args.beta_margin,
        decision_distance=args.decision_distance,
        intent_window=args.intent_window,
        type_probs=parse_type_probs(args.type_probs),
        courtesy_grace_steps=args.courtesy_grace_steps,
        time_penalty=args.time_penalty)
    env = SumoPlanningEnv(cfg, label=f"branch_oracle_{args.scenario}",
                          continuous=True)
    if probes is None:
        probes = [float(a) for a in env.cfg.discrete_actions]
    rows = {"branch_oracle": [], "time_gap": []}
    try:
        for ep in range(args.episodes):
            seed = 200_000 + ep
            for name, runner in (
                    ("branch_oracle",
                     lambda: run_branch_oracle(
                         env, seed, probes, args.commit_steps,
                         args.exit_distance, args.rollout_steps)),
                    ("time_gap", lambda: run_time_gap(env, seed))):
                ret, info = runner()
                rows[name].append({
                    "seed": seed,
                    "return": ret,
                    "outcome": info["outcomes"][0] or "timeout",
                })
    finally:
        env.close()

    result = {
        "config": vars(args),
        "branch_oracle": summarize(rows["branch_oracle"]),
        "time_gap": summarize(rows["time_gap"]),
        "rows": rows,
    }
    print(json.dumps(result, indent=2))
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
