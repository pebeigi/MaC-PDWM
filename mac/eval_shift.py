"""Evaluate saved planners under distribution shift (no retraining).

Train-time policies are loaded from ``policy_*.pt`` next to their metrics
files and rolled out in an environment whose traffic density, channel
strength, type mix, or scenario differs from training.
"""
import argparse
import glob
import json
import os

import torch

from mac.agents.ppo import PPO
from mac.envs.sumo_planning_env import EnvConfig, SumoPlanningEnv, parse_type_probs
from mac.models.belief import parse_probe_accels
from mac.train_planner import BeliefAugmentedEnv, build_encoder, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner_dir", default="data/mac/planner_cross")
    parser.add_argument("--out_dir", default="data/mac/planner_shift")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--bg_scale", type=float, default=2.5)
    parser.add_argument("--beta_intent", type=float, default=0.90)
    parser.add_argument("--type_probs", default="0.15,0.35,0.50")
    parser.add_argument("--scenario", default="cross")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    policies = sorted(glob.glob(os.path.join(args.planner_dir, "policy_*.pt")))
    if not policies:
        raise SystemExit(f"no policy_*.pt in {args.planner_dir}")

    for path in policies:
        tag = os.path.basename(path)[len("policy_"):-len(".pt")]
        metrics_path = os.path.join(args.planner_dir, f"metrics_{tag}.json")
        config = {}
        if os.path.isfile(metrics_path):
            with open(metrics_path) as handle:
                config = json.load(handle).get("config", {})
        belief = config.get("belief", tag.split("_")[0])
        world_model = config.get("world_model", "data/mac/world_model_cross.pt")
        probes = config.get("probes", "")
        n_samples = int(config.get("n_samples", 8))
        sample_steps = int(config.get("sample_steps", 10))

        class _Args:
            pass
        enc_args = _Args()
        enc_args.world_model = world_model
        enc_args.n_samples = n_samples
        enc_args.sample_steps = sample_steps
        enc_args.probes = probes

        cfg = EnvConfig(
            scenario=args.scenario, seed=0, horizon=150,
            bg_rate_scale=args.bg_scale, beta_intent=args.beta_intent,
            type_probs=parse_type_probs(args.type_probs),
        )
        env = SumoPlanningEnv(cfg, label=f"shift_{tag}")
        try:
            encoder = build_encoder(belief, enc_args, env, device)
            wrapper = BeliefAugmentedEnv(
            env, encoder, belief,
            commit_steps=int(config.get("commit_steps", 1) or 1),
            plan_action=bool(config.get("plan_action", False)),
            probe_accels=parse_probe_accels(probes),
        )
            agent = PPO(wrapper.obs_dim, wrapper.n_actions, device)
            state = torch.load(path, map_location=device, weights_only=False)
            if "policy" in state:
                agent.load_state_dict(state)
            else:  # checkpoints written before observation normalisation
                agent.policy.load_state_dict(state)
                agent.obs_norm = None
            metrics = evaluate(wrapper, agent, args.episodes, seed_offset=200000)
        finally:
            env.close()

        blob = {
            "config": {
                **config,
                "belief": belief,
                "eval_bg_scale": args.bg_scale,
                "eval_beta_intent": args.beta_intent,
                "eval_type_probs": args.type_probs,
                "eval_scenario": args.scenario,
                "shift": True,
            },
            "history": [{**metrics, "iteration": 0}],
        }
        out = os.path.join(args.out_dir, f"metrics_{tag}.json")
        with open(out, "w") as handle:
            json.dump(blob, handle, indent=2)
        print(f"[{tag}] belief={belief} probes={parse_probe_accels(probes)} "
              f"succ={metrics['success_rate']:.3f} coll={metrics['collision_rate']:.3f} "
              f"ret={metrics['return']:.2f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
