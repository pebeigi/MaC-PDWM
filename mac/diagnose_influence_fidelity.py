"""Does the learned influence estimate track the simulator's true one?

At every step this queries the diffusion belief encoder and the oracle encoder on
the *same* state and compares their counterfactual contrast in P(yield). Feature
scale and Monte-Carlo noise are separate problems; this measures whether the
intention head has learned the channel at all. If the correlation is near zero,
no amount of downstream tuning lets the policy negotiate, and the fix belongs in
world-model training rather than in the planner.
"""
import argparse

import numpy as np
import torch

from mac.envs.sumo_planning_env import EnvConfig, SumoPlanningEnv, parse_type_probs
from mac.train_planner import BeliefAugmentedEnv, build_encoder

P_YIELD_SLOT = 4
N_PROBE_FEATURES = 11


def contrast(vec, n_probes):
    """assert-minus-yield in P(yield), read straight out of the belief vector."""
    last = (n_probes - 1) * N_PROBE_FEATURES + P_YIELD_SLOT
    return float(vec[last] - vec[P_YIELD_SLOT])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-model", default="data/mac/world_model_cross.pt")
    ap.add_argument("--scenario", default="cross")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--sample-steps", type=int, default=10)
    ap.add_argument("--probes", default="")
    ap.add_argument("--beta_intent", type=float, default=1.5)
    ap.add_argument("--beta_margin", type=float, default=0.6)
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
    env = SumoPlanningEnv(cfg, label="fidelity")
    encoders = {name: build_encoder(name, args, env, device)
                for name in ("diffusion", "history", "oracle")}
    n_probes = len(encoders["oracle"].probe_accels)
    wrapper = BeliefAugmentedEnv(env, encoders["oracle"], "oracle")

    rng = np.random.default_rng(0)
    rows = {k: [] for k in encoders}
    try:
        for ep in range(args.episodes):
            wrapper.reset(seed=50_000 + ep)
            done = False
            while not done:
                ego = env.egos[0]
                state = env._last_snapshot.get(ego.veh_id)
                if state is not None and not ego.done:
                    for name, enc in encoders.items():
                        rows[name].append(
                            contrast(enc(env.frames, ego.veh_id, state["speed"]),
                                     n_probes))
                    _, _, done, _ = wrapper.step(int(rng.integers(wrapper.n_actions)))
                else:
                    _, _, done, _ = wrapper.step(0)
    finally:
        env.close()

    truth = np.asarray(rows["oracle"])
    live = np.abs(truth) > 1e-3
    print(f"{len(truth)} steps, channel live on {live.mean():.1%}")
    print(f"true contrast on live steps: mean={truth[live].mean():+.3f} "
          f"sd={truth[live].std():.3f}")
    for name in ("diffusion", "history"):
        est = np.asarray(rows[name])
        print(f"\n{name}: mean={est.mean():+.3f} sd={est.std():.3f}")
        if est.std() < 1e-9:
            print("  contrast is identically zero (no plan-conditioning by construction)")
            continue
        r_all = np.corrcoef(est, truth)[0, 1]
        r_live = np.corrcoef(est[live], truth[live])[0, 1] if live.sum() > 2 else np.nan
        # Slope of truth on estimate: 1.0 would be perfectly calibrated.
        slope = np.polyfit(est[live], truth[live], 1)[0] if live.sum() > 2 else np.nan
        print(f"  corr with truth: all steps={r_all:+.3f}  live steps={r_live:+.3f}")
        print(f"  regression slope truth~est on live steps: {slope:+.3f}")
        # Sign agreement is what a policy would actually act on.
        agree = (np.sign(est[live]) == np.sign(truth[live])).mean()
        print(f"  sign agreement on live steps: {agree:.1%}")


if __name__ == "__main__":
    main()
