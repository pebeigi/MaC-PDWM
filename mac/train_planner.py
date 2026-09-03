"""Train the negotiation-aware planner.

Five configurations share this script so the ablation is exact:

  --belief none      PPO on raw observations
  --belief geometry  PPO plus ego-plan vs constant-velocity neighbour risk
  --belief kernel    geometry risk plus an analytic channel fitted offline
  --belief oracle    the same, but reading the simulator's true channel
  --belief mean      PPO plus a deterministic single-future forecast
  --belief diffusion PPO plus the diffusion belief over counterfactual futures
  --belief history   PPO plus a history-only world model (plan input zeroed)
"""
import argparse
import collections
import json
import os
import time

import numpy as np
import torch

from mac.agents.ppo import PPO, RolloutBuffer
from mac.envs.sumo_planning_env import EnvConfig, SumoPlanningEnv, parse_type_probs
from mac.fit_kernel import MARGIN_CLIP
from mac.models.belief import DEFAULT_PROBE_ACCELS, BeliefEncoder, parse_probe_accels
from mac.models.diffusion_world_model import DiffusionWorldModel


def load_guidance(path, device):
    """Guidance weight calibrated on held-out open-loop episodes, 1.0 if unset."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return float(ckpt.get("guidance", 1.0))


def load_world_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = DiffusionWorldModel(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["pos_scale"], checkpoint["vel_scale"]


class BeliefAugmentedEnv:
    """Wraps the planning env and appends belief features to the observation."""

    def __init__(self, env, encoder, mode, commit_steps=1, plan_action=False,
                 probe_accels=None):
        self.env = env
        self.encoder = encoder
        self.mode = mode
        self.plan_action = bool(plan_action)
        self.probe_accels = tuple(
            DEFAULT_PROBE_ACCELS if probe_accels is None else probe_accels)
        extra = 0 if encoder is None else encoder.feature_dim
        self.obs_dim = env.obs_dim + extra
        if self.plan_action:
            self.n_actions = len(self.probe_accels)
            future_len = int(getattr(encoder, "future_len", 20) or 20) if encoder else 20
            # Action *is* the F-step probe; hold it for the queried horizon.
            self.commit_steps = future_len if int(commit_steps) <= 1 else max(1, int(commit_steps))
        else:
            self.commit_steps = max(1, int(commit_steps))
            self.n_actions = len(env.cfg.discrete_actions)

    def _augment(self, obs):
        if self.encoder is None:
            return obs[0]
        ego = self.env.egos[0]
        state = self.env._last_snapshot.get(ego.veh_id)
        if state is None or ego.done:
            belief = self.encoder.zero_features()
        else:
            belief = self.encoder(self.env.frames, ego.veh_id, state["speed"])
        return np.concatenate([obs[0], belief]).astype(np.float32)

    def reset(self, seed=None):
        return self._augment(self.env.reset(seed=seed))

    def step(self, action):
        total = 0.0
        info = {}
        done = False
        obs = None
        if self.plan_action:
            accel = float(self.probe_accels[int(action)])
            env_action = int(np.argmin(np.abs(
                np.asarray(self.env.cfg.discrete_actions) - accel)))
        else:
            env_action = int(action)
        for _ in range(self.commit_steps):
            obs, rewards, dones, info = self.env.step(np.array([env_action]))
            total += float(rewards[0])
            done = bool(dones[0])
            if done:
                break
        return self._augment(obs), total, done, info


def make_oracle_fn(env, horizon=10):
    """Per-neighbour analytic P(yield) approximation to the simulator channel.

    Decided drivers report what they decided. For undecided reactive drivers the
    kernel is rolled forward rather than evaluated at the probe acceleration
    directly: the simulator reads intent from an exponential moving average of
    past ego acceleration, which approaches a newly chosen acceleration
    geometrically, and the priority margin moves as the probe changes the ego's
    own arrival time. Substituting the probe for the EMA instead would credit the
    ego with instantaneous, complete influence and make the oracle an
    over-confident belief rather than a ceiling. The commitment step is unknown,
    so the kernel is averaged over the horizon the probe covers, which is the
    same window the world model is supervised on.
    """
    def oracle(neighbor_ids, probe_accel):
        drivers = env.drivers
        snapshot = env._last_snapshot or {}
        ego_states = env._ego_states()
        ego_state = ego_states[0] if ego_states else None
        dt = env.dt
        max_speed = env.spec.max_speed
        alpha = float(np.clip(dt / max(drivers.intent_window, dt), 0.0, 1.0))
        ema0 = float(drivers._ego_accel_ema)
        steps = np.arange(1, horizon + 1)
        # EMA after n steps of holding the probe acceleration.
        ema = (1.0 - alpha) ** steps * ema0 + (1.0 - (1.0 - alpha) ** steps) * probe_accel

        out = np.zeros((len(neighbor_ids), 2), dtype=np.float32)
        for k, veh_id in enumerate(neighbor_ids):
            decided = drivers.resolved.get(veh_id)
            driver_type = drivers.types.get(veh_id)
            if decided == "yield" or driver_type == "yielder":
                out[k] = (1.0, 0.0)
                continue
            if decided == "contest" or driver_type == "contester":
                out[k] = (0.0, 1.0)
                continue
            state = snapshot.get(veh_id)
            if state is None or ego_state is None or driver_type is None:
                out[k] = (0.5, 0.5)
                continue
            d_other = drivers._signed_distance(state["x"], state["y"], state["heading"])
            other_speed = max(state["speed"], 0.5)
            other_ttc = (d_other - other_speed * dt * steps) / other_speed

            t = dt * steps
            ego_speed = np.clip(ego_state["speed"] + probe_accel * t, 0.0, max_speed)
            travelled = ego_state["speed"] * t + 0.5 * probe_accel * t**2
            ego_ttc = (ego_state["d_conflict"] - travelled) / np.maximum(ego_speed, 0.5)

            logit = (drivers.beta_margin * (other_ttc - ego_ttc)
                     + drivers.beta_intent * ema + drivers.beta_bias)
            # Marginalise the unobservable per-driver offset, matching the
            # channel this feature is meant to approximate.
            scale = np.sqrt(1.0 + np.pi * drivers.stubbornness_scale ** 2 / 8.0)
            p = float(np.mean(1.0 / (1.0 + np.exp(-logit / scale))))
            out[k] = (p, 1.0 - p)
        return out
    return oracle


def make_fitted_kernel_fn(env, params, horizon=10):
    """Analytic P(yield) from coefficients fitted offline by ``mac.fit_kernel``.

    Unlike ``make_oracle_fn`` this reads nothing from ``env.drivers``: no true
    coefficients, no latent ``types``, no ``resolved`` decisions, and not the
    simulator's own intent EMA. Everything comes from the observable snapshot,
    the known conflict point, and an EMA of past ego acceleration that this
    closure maintains itself, so the ``kernel`` arm is a baseline a deployed
    system could actually build rather than a privileged ceiling.
    """
    beta_margin = float(params["beta_margin"])
    beta_intent = float(params["beta_intent"])
    beta_bias = float(params["beta_bias"])
    window = float(params["intent_window"])
    horizon = int(params.get("horizon", horizon))
    point = np.asarray(env.spec.conflict_point, dtype=float)
    alpha = float(np.clip(env.dt / max(window, env.dt), 0.0, 1.0))
    # Mutable so the EMA survives across calls; one update per simulator step,
    # not per probe, or the average would advance len(probes) times too fast.
    state = {"ema": 0.0, "speed": None, "step": -1}

    def signed_ttc(x, y, speed, heading):
        approach = point - np.asarray([x, y], dtype=float)
        forward = np.asarray([np.cos(heading), np.sin(heading)])
        sign = 1.0 if float(np.dot(approach, forward)) >= 0 else -1.0
        return sign * float(np.linalg.norm(approach)) / max(float(speed), 0.5)

    def kernel(neighbor_ids, probe_accel):
        snapshot = env._last_snapshot or {}
        out = np.full((len(neighbor_ids), 2), 0.5, dtype=np.float32)
        # _ego_states() drops the vehicle id, and the kernel needs the pose, so
        # the snapshot is indexed directly by the first ego still in play.
        ego = next((snapshot[e.veh_id] for e in env.egos
                    if not e.done and e.veh_id in snapshot), None)
        if ego is None:
            return out
        ego_speed = float(ego["speed"])

        if env.step_count != state["step"]:
            if state["speed"] is not None:
                observed = (ego_speed - state["speed"]) / env.dt
                state["ema"] = (1.0 - alpha) * state["ema"] + alpha * observed
            state["speed"] = ego_speed
            state["step"] = env.step_count

        # Roll the intent signal forward under the probe and average over the
        # horizon, matching the feature ``mac.fit_kernel`` was fitted on.
        steps = np.arange(1, horizon + 1)
        ema = ((1.0 - alpha) ** steps * state["ema"]
               + (1.0 - (1.0 - alpha) ** steps) * probe_accel)
        intent = float(np.mean(ema))

        ego_ttc = signed_ttc(ego["x"], ego["y"], ego_speed, ego["heading"])
        for k, veh_id in enumerate(neighbor_ids):
            other = snapshot.get(veh_id)
            if other is None:
                continue
            margin = signed_ttc(other["x"], other["y"], other["speed"],
                                other["heading"]) - ego_ttc
            margin = float(np.clip(margin, -MARGIN_CLIP, MARGIN_CLIP))
            logit = beta_margin * margin + beta_intent * intent + beta_bias
            p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
            out[k] = (p, 1.0 - p)
        return out

    return kernel


def load_kernel_params(path):
    with open(path) as handle:
        return json.load(handle)


def build_encoder(mode, args, env, device):
    if mode == "none":
        return None
    probes = parse_probe_accels(getattr(args, "probes", None))
    blocks = [
        value.strip() for value in
        str(getattr(args, "belief_blocks", "")).split(",")
        if value.strip()]
    if mode in ("geometry", "kernel", "oracle"):
        # "kernel" is the fair baseline: coefficients fitted offline from the
        # same data the world model sees. "oracle" keeps the privileged form
        # that reads the simulator's channel, as a diagnostic ceiling.
        kernel_params = getattr(args, "kernel_params", "") or ""
        if mode == "kernel":
            if not kernel_params:
                raise SystemExit(
                    "--kernel_params is required for the kernel arm: fit it with "
                    "`python -m mac.fit_kernel --data data/mac/<scene>.npz "
                    "--out data/mac/kernel_<scene>.json`. Use --belief oracle for "
                    "the privileged channel instead.")
            oracle_fn = make_fitted_kernel_fn(env, load_kernel_params(kernel_params))
        elif mode == "oracle":
            oracle_fn = make_oracle_fn(env)
        else:
            oracle_fn = None
        encoder = BeliefEncoder(
            None, device, env.dt, env.spec.max_speed, 1.0, 1.0,
            n_samples=1, sample_steps=1, mode=mode, probe_accels=probes,
            history_len=5, future_len=20, n_neighbors=env.cfg.n_neighbors,
            oracle_fn=oracle_fn,
            conflict_point=env.spec.conflict_point,
            approach_edge_prefixes=env.spec.approach_edge_prefixes,
            decision_distance=env.cfg.decision_distance,
        )
        return encoder.set_feature_blocks(blocks) if blocks else encoder
    model, pos_scale, vel_scale = load_world_model(args.world_model, device)
    # Both belief modes draw the same number of futures at the same cost; only
    # "mean" collapses them, so the ablation isolates multi-hypothesis structure
    # rather than sample budget.
    encoder = BeliefEncoder(
        model, device, env.dt, env.spec.max_speed, pos_scale, vel_scale,
        n_samples=args.n_samples, sample_steps=args.sample_steps, mode=mode,
        probe_accels=probes, guidance=load_guidance(args.world_model, device),
        conflict_point=env.spec.conflict_point,
        approach_edge_prefixes=env.spec.approach_edge_prefixes,
        decision_distance=env.cfg.decision_distance,
    )
    return encoder.set_feature_blocks(blocks) if blocks else encoder


def evaluate(wrapper, agent, episodes, seed_offset=100000):
    outcomes = collections.Counter()
    returns, steps, brakes, success_steps = [], [], [], []
    # Episodes in which at least one reactive driver committed while the ego was
    # in play are the only ones where influence can pay; report them separately
    # so an aggregate average cannot hide (or manufacture) a negotiation effect.
    infl_returns, infl_success, infl_collision, infl_yields = [], 0, 0, []
    collision_by_type = collections.Counter()
    collision_by_resolved = collections.Counter()
    for ep in range(episodes):
        # The world-model arms draw belief samples with torch randomness. Without
        # a per-episode seed they face noise that none/geometry/oracle do not,
        # and repeated evaluations of the same policy disagree.
        torch.manual_seed(seed_offset + ep)
        obs = wrapper.reset(seed=seed_offset + ep)
        total = 0.0
        for _ in range(wrapper.env.cfg.horizon):
            action, _, _ = agent.select(obs, deterministic=True)
            obs, reward, done, info = wrapper.step(action)
            total += reward
            if done:
                break
        outcome = info["outcomes"][0] or "timeout"
        outcomes[outcome] += 1
        if outcome == "collision":
            partner_types = info.get("collision_partner_type") or [""]
            partner_res = info.get("collision_partner_resolved") or [""]
            collision_by_type[partner_types[0] or "unknown"] += 1
            collision_by_resolved[partner_res[0] or "unresolved"] += 1
        returns.append(total)
        steps.append(info["step"])
        stats = info["driver_stats"]
        brakes.append(stats["ego_induced_brakes"])
        if outcome == "success":
            success_steps.append(info["step"])
        n_opportunities = stats.get(
            "reactive_opportunities", stats.get("reactive_resolved", 0))
        n_reactive = stats.get("reactive_resolved", 0)
        if n_opportunities > 0:
            infl_returns.append(total)
            infl_success += int(outcome == "success")
            infl_collision += int(outcome == "collision")
            infl_yields.append(
                stats.get("reactive_yields", 0) / max(n_reactive, 1))
    n_infl = len(infl_returns)
    return {
        "return": float(np.mean(returns)),
        "steps": float(np.mean(steps)),
        "crossing_steps": float(np.mean(success_steps)) if success_steps else float("nan"),
        "induced_brakes": float(np.mean(brakes)),
        "success_rate": outcomes["success"] / episodes,
        "collision_rate": outcomes["collision"] / episodes,
        "timeout_rate": outcomes["timeout"] / episodes,
        "lost_rate": outcomes["lost"] / episodes,
        # Influenceable subset.
        "n_influenceable": n_infl,
        "return_infl": float(np.mean(infl_returns)) if n_infl else float("nan"),
        "success_rate_infl": infl_success / n_infl if n_infl else float("nan"),
        "collision_rate_infl": infl_collision / n_infl if n_infl else float("nan"),
        "reactive_yield_rate": float(np.mean(infl_yields)) if n_infl else float("nan"),
        "collision_by_type": dict(collision_by_type),
        "collision_by_resolved": dict(collision_by_resolved),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--belief",
                        choices=["none", "geometry", "kernel", "oracle",
                                 "mean", "diffusion", "history"],
                        default="diffusion")
    parser.add_argument("--world_model", default="data/mac/world_model_cross.pt")
    parser.add_argument("--kernel_params", default="",
                        help="JSON from mac.fit_kernel; required by --belief kernel")
    parser.add_argument("--scenario", default="cross")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--steps_per_iter", type=int, default=2048)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--probes", default="",
                        help="comma-separated constant-accel probes, brake→assert. "
                             "Empty = {-4,0,+3} (on the discrete action set).")
    parser.add_argument(
        "--belief_blocks", default="",
        help="comma-separated risk,clear,spread,intent,shift,waypoint; "
             "empty enables all blocks")
    parser.add_argument("--commit_steps", type=int, default=1,
                        help="hold the chosen accel for this many env steps before "
                             "the next PPO decision (receding-horizon commit).")
    parser.add_argument("--plan_action", action="store_true",
                        help="PPO chooses among probe plans {yield,hold,assert} and "
                             "commits that accel for F steps. This is the negotiation "
                             "object, not a one-step a_t.")
    parser.add_argument("--bg_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default="data/mac/planner")
    parser.add_argument("--tag", default=None, help="overrides the run name used for outputs")
    # Exploration has to stay alive long enough for the agent to discover that
    # crossing beats waiting; a fixed low coefficient collapses the policy onto
    # the creeping local optimum before it ever succeeds.
    parser.add_argument("--entropy_start", type=float, default=0.03)
    parser.add_argument("--entropy_end", type=float, default=0.005)
    # 0.0 severs the influence channel: latent types stop depending on ego motion.
    parser.add_argument("--beta_intent", type=float, default=2.5)
    parser.add_argument("--beta_margin", type=float, default=0.3,
                        help="weight on the priority (TTC) term of the channel")
    parser.add_argument("--decision_distance", type=float, default=None,
                        help="commit radius; default = scenario-specific")
    parser.add_argument("--intent_window", type=float, default=0.8,
                        help="EMA window of the ego-motion intent signal (s)")
    parser.add_argument("--type_probs", default="0.1,0.1,0.8",
                        help="yielder,contester,reactive probabilities")
    parser.add_argument("--courtesy_weight", type=float, default=0.15)
    parser.add_argument("--courtesy_grace_steps", type=int, default=3,
                        help="steps after a concession that are not charged to the ego")
    parser.add_argument("--time_penalty", type=float, default=0.06,
                        help="per-step cost of not having crossed yet; sets how "
                             "expensive patience is relative to collision risk")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    cfg = EnvConfig(scenario=args.scenario, seed=args.seed, horizon=150,
                    bg_rate_scale=args.bg_scale, beta_intent=args.beta_intent,
                    beta_margin=args.beta_margin,
                    decision_distance=args.decision_distance,
                    intent_window=args.intent_window,
                    type_probs=parse_type_probs(args.type_probs),
                    courtesy_weight=args.courtesy_weight,
                    courtesy_grace_steps=args.courtesy_grace_steps,
                    time_penalty=args.time_penalty)
    env = SumoPlanningEnv(cfg, label=f"planner_{args.belief}_{args.seed}")
    probes = parse_probe_accels(args.probes)
    encoder = build_encoder(args.belief, args, env, device)
    wrapper = BeliefAugmentedEnv(
        env, encoder, args.belief, commit_steps=args.commit_steps,
        plan_action=args.plan_action, probe_accels=probes,
    )

    # Preserve the original per-environment-step discount when one policy
    # transition commits several low-level actions.
    agent = PPO(
        wrapper.obs_dim, wrapper.n_actions, device,
        gamma=0.99 ** wrapper.commit_steps)
    buffer = RolloutBuffer()

    tag = args.tag or f"{args.belief}_seed{args.seed}"
    os.makedirs(args.out_dir, exist_ok=True)
    history = []

    print(f"[{tag}] obs_dim={wrapper.obs_dim} actions={wrapper.n_actions} "
          f"probes={probes} commit={wrapper.commit_steps} "
          f"plan_action={args.plan_action} device={device}")

    obs = wrapper.reset()
    episode_return, episode_returns = 0.0, collections.deque(maxlen=50)
    outcomes = collections.Counter()

    try:
        for iteration in range(args.iterations):
            started = time.time()
            frac = iteration / max(args.iterations - 1, 1)
            agent.entropy_coef = args.entropy_start + frac * (args.entropy_end - args.entropy_start)
            buffer.clear()
            for _ in range(args.steps_per_iter):
                action, logprob, value = agent.select(obs)
                next_obs, reward, done, info = wrapper.step(action)
                buffer.add(obs, action, logprob, reward, value, done)
                episode_return += reward
                obs = next_obs
                if done:
                    outcomes[info["outcomes"][0] or "timeout"] += 1
                    episode_returns.append(episode_return)
                    episode_return = 0.0
                    obs = wrapper.reset()

            _, _, last_value = agent.select(obs)
            stats = agent.update(buffer, last_value)

            total = sum(outcomes.values()) or 1
            line = (f"[{tag}] iter {iteration + 1:3d}/{args.iterations} "
                    f"return={np.mean(episode_returns) if episode_returns else 0:6.2f} "
                    f"succ={outcomes['success'] / total:.2f} "
                    f"coll={outcomes['collision'] / total:.2f} "
                    f"tout={outcomes['timeout'] / total:.2f} "
                    f"ent={stats['entropy']:.3f} (ec={agent.entropy_coef:.3f}) "
                    f"({time.time() - started:.0f}s)")
            print(line, flush=True)
            outcomes.clear()

            if (iteration + 1) % args.eval_every == 0 or iteration == args.iterations - 1:
                metrics = evaluate(wrapper, agent, args.eval_episodes)
                metrics["iteration"] = iteration + 1
                history.append(metrics)
                print(f"[{tag}] EVAL {json.dumps(metrics)}", flush=True)
                with open(os.path.join(args.out_dir, f"metrics_{tag}.json"), "w") as handle:
                    json.dump({"config": vars(args), "history": history}, handle, indent=2)
                torch.save(agent.state_dict(),
                           os.path.join(args.out_dir, f"policy_{tag}.pt"))
                obs = wrapper.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
