"""How much do belief features actually move across probes?

If plan-conditioning is usable, P(yield) and risk should shift when the probe
goes from yield to assert. A history-only encoder should show near-zero
intention contrast (same neighbour prediction) while still moving risk through
ego geometry alone.
"""
import argparse

import numpy as np
import torch

from mac.data.normalize import POS_SCALE, VEL_SCALE
from mac.envs.sumo_planning_env import EnvConfig, SumoPlanningEnv
from mac.models.belief import BeliefEncoder, parse_probe_accels
from mac.models.diffusion_world_model import DiffusionWorldModel
from mac.policies.rule_based import TimeGapPolicy


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = DiffusionWorldModel(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["pos_scale"], checkpoint["vel_scale"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_model", default="data/mac/world_model_cross.pt")
    parser.add_argument("--history_model",
                        default="data/mac/world_model_history_cross.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--bg_scale", type=float, default=1.5)
    parser.add_argument("--beta_intent", type=float, default=0.45)
    parser.add_argument("--probes", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    probes = parse_probe_accels(args.probes)

    device = torch.device(args.device)
    cfg = EnvConfig(scenario="cross", seed=0, horizon=80, bg_rate_scale=args.bg_scale,
                    beta_intent=args.beta_intent)
    env = SumoPlanningEnv(cfg, label="belief_sens")
    policy = TimeGapPolicy(env)

    models = {}
    for tag, path in (("diffusion", args.world_model), ("history", args.history_model)):
        if not path:
            continue
        try:
            model, pos, vel = load_model(path, device)
        except FileNotFoundError:
            print(f"skip {tag}: {path} missing")
            continue
        models[tag] = BeliefEncoder(model, device, env.dt, env.spec.max_speed, pos, vel,
                                    n_samples=8, sample_steps=10, mode=tag,
                                    probe_accels=probes)

    contrasts = {tag: [] for tag in models}
    try:
        for ep in range(args.episodes):
            obs = env.reset(seed=1000 + ep)
            policy.reset()
            for _ in range(cfg.horizon):
                ego = env.egos[0]
                state = env._last_snapshot.get(ego.veh_id)
                if state is not None and not ego.done and len(env.frames) >= 5:
                    for tag, encoder in models.items():
                        feat = encoder(env.frames, ego.veh_id, state["speed"])
                        # Last five dimensions are assert-minus-yield contrasts.
                        contrasts[tag].append(feat[-5:])
                obs, _, dones, _ = env.step(policy.act(obs))
                if bool(np.all(dones)):
                    break
    finally:
        env.close()

    names = ("d_risk", "d_clear", "d_spread", "d_P(yield)", "d_shift")
    print(f"probes {probes}; contrast = last − first")
    for tag, rows in contrasts.items():
        arr = np.asarray(rows)
        print(f"\n{tag}: n={len(arr)}")
        for i, name in enumerate(names):
            print(f"  {name:12s}  mean={arr[:, i].mean():+.4f}  "
                  f"|mean|={np.abs(arr[:, i]).mean():.4f}  std={arr[:, i].std():.4f}")


if __name__ == "__main__":
    main()
