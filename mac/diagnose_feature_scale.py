"""Per-dimension scale of raw and belief inputs before PPO running normalisation."""
import argparse

import numpy as np
import torch

from mac.envs.sumo_planning_env import EnvConfig, SumoPlanningEnv, parse_type_probs
from mac.train_planner import BeliefAugmentedEnv, build_encoder

PROBE_FEATURE_NAMES = ("risk", "mean_clear", "min_clear", "spread", "p_yield",
                       "p_contest", "shift", "mid_x", "mid_y", "end_x", "end_y")
CONTRAST_NAMES = ("d_risk", "d_clear", "d_spread", "d_p_yield", "d_shift")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--belief", default="diffusion")
    ap.add_argument("--world-model", default="data/mac/world_model_cross.pt")
    ap.add_argument("--scenario", default="cross")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--sample-steps", type=int, default=10)
    ap.add_argument("--conflict-radius", type=float, default=10.0)
    ap.add_argument("--probe-accels", default="-4,0,2")
    ap.add_argument("--commit-steps", type=int, default=1)
    # Must match the channel the experiments actually run with, otherwise the
    # measured influence belongs to a different environment.
    ap.add_argument("--beta_intent", type=float, default=2.5)
    ap.add_argument("--beta_margin", type=float, default=0.3)
    ap.add_argument("--decision_distance", type=float, default=None)
    ap.add_argument("--intent_window", type=float, default=0.8)
    ap.add_argument("--type_probs", default="0.1,0.1,0.8")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = EnvConfig(scenario=args.scenario, gui=False,
                    beta_intent=args.beta_intent, beta_margin=args.beta_margin,
                    decision_distance=args.decision_distance,
                    intent_window=args.intent_window,
                    type_probs=parse_type_probs(args.type_probs))
    env = SumoPlanningEnv(cfg, label=f"scalediag_{args.belief}")
    encoder = build_encoder(args.belief, args, env, device)
    wrapper = BeliefAugmentedEnv(env, encoder, args.belief,
                                 commit_steps=args.commit_steps)

    rng = np.random.default_rng(0)
    rows = []
    for ep in range(args.episodes):
        obs = wrapper.reset(seed=10_000 + ep)
        done = False
        while not done:
            rows.append(obs.copy())
            obs, _, done, _ = wrapper.step(int(rng.integers(wrapper.n_actions)))
    env.close()

    x = np.asarray(rows)
    n_raw = env.obs_dim
    raw, belief = x[:, :n_raw], x[:, n_raw:]
    print(f"{len(x)} steps   raw dims={n_raw}   belief dims={belief.shape[1]}")
    print(f"raw obs   |mean|={np.abs(raw).mean():.3f}  sd(per-dim, median)="
          f"{np.median(raw.std(axis=0)):.3f}  sd max={raw.std(axis=0).max():.3f}")
    if belief.shape[1] == 0:
        return

    print(f"belief    |mean|={np.abs(belief).mean():.3f}  sd(per-dim, median)="
          f"{np.median(belief.std(axis=0)):.4f}")
    accels = [float(a) for a in args.probe_accels.split(",")]
    n_probe = len(PROBE_FEATURE_NAMES)
    print("\n  feature            " + "".join(f"a={a:<8.0f}" for a in accels))
    for f, name in enumerate(PROBE_FEATURE_NAMES):
        sds = [belief[:, p * n_probe + f].std() for p in range(len(accels))]
        print(f"  {name:<18} " + "".join(f"{s:<10.4f}" for s in sds))
    base = len(accels) * n_probe
    print("\n  contrast (assert - yield):")
    for c, name in enumerate(CONTRAST_NAMES):
        col = belief[:, base + c]
        print(f"  {name:<18} sd={col.std():.4f}  mean={col.mean():+.4f}")

    # How often is the channel actually live, and how strong is it then? An
    # average over all steps is diluted by neighbours that already committed,
    # for whom no probe can change anything.
    live = np.abs(belief[:, base + 3]) > 1e-3
    print(f"\n  steps where P(yield) responds to the probe: {live.mean():.1%}")
    if live.any():
        v = belief[live, base + 3]
        q = np.percentile(np.abs(v), [50, 90, 99])
        print(f"  |d_p_yield| on those steps: mean={np.abs(v).mean():.3f}  "
              f"median={q[0]:.3f}  p90={q[1]:.3f}  p99={q[2]:.3f}  max={np.abs(v).max():.3f}")


if __name__ == "__main__":
    main()
