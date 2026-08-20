"""Belief features derived from the diffusion world model.

At every decision step the ego queries the world model with a small set of
constant-acceleration plans (multi-step roll-outs of one-step actions) and
summarises the predicted neighbour responses. The policy sees this summary,
not the full sample tensor, but the summary is no longer just a collision
margin: it includes predicted neighbour waypoints and the displacement of
those waypoints when the probe changes (zero unless y moves with u).

A usable closed-loop gap over a history-only encoder is evidence that the
policy exploits plan-conditioned predictions. Risk features that move with the
probe *ego geometry* alone can still support ordinary risk-aware planning.
"""
import numpy as np
import torch

from mac.data.normalize import decode_samples
from mac.data.scene import ego_yaw_rate, extract_scene, synthetic_plan

# Default probes sit on the discrete action set {-4,-2,0,1,2,3} so every
# counterfactual query is on the randomised support (yield / hold / assert).
# The assert probe reaches the top of the action set: how far apart the probes
# are decides how many commitments they separate, as much as how far ahead they
# reach.
DEFAULT_PROBE_ACCELS = (-4.0, 0.0, 3.0)
ACTION_ALIGNED_PROBES = (-4.0, 0.0, 3.0)
ACTION_PROBES = (-4.0, -2.0, 0.0, 1.0, 2.0, 3.0)
LEGACY_PROBE_ACCELS = (-3.0, 0.0, 2.0)  # off-support; kept for old checkpoints
PROBE_ACCELS = DEFAULT_PROBE_ACCELS  # back-compat alias

RELEVANCE_TAU = 10.0


def parse_probe_accels(text=None, default=DEFAULT_PROBE_ACCELS):
    """Parse a comma-separated acceleration list, ordered brake → assert."""
    if text is None or str(text).strip() == "":
        return tuple(float(a) for a in default)
    vals = tuple(float(x.strip()) for x in str(text).split(",") if x.strip())
    if len(vals) < 2:
        raise ValueError("need at least two probe accelerations (yield and assert)")
    return vals


class BeliefEncoder:
    """Summarises counterfactual world-model queries into a fixed-size vector.

    ``mode="mean"`` collapses the drawn futures into their pointwise average
    before computing any statistic, which is the certainty-equivalence ablation:
    it keeps the plan-conditioning and the model capacity but destroys the
    multi-hypothesis structure, so the spread feature is identically zero.

    ``mode="history"`` feeds a zero plan to the world model for every probe, so
    the predictor is $p(y \\mid h)$ rather than $p(y \\mid h, u)$. Clearance
    statistics are still computed against each probe's ego trajectory, so the
    features remain plan-dependent through geometry alone; what is removed is
    plan-conditioning in the predicted neighbour response.

    ``mode="geometry"`` never queries a world model. Neighbours are rolled out
    with constant velocity; risk/clearance are ego-plan vs that CV forecast.
    This isolates engineered conflict geometry from learned prediction.

    ``mode="kernel"`` uses the same CV geometry but fills the intention slots
    from an analytic approximation to the simulator channel. It is a diagnostic,
    not an upper bound; the exact non-causal bound is ``mac.eval_branch_oracle``.
    ``mode="oracle"`` remains as a backward-compatible alias.
    """

    def __init__(self, world_model, device, dt, max_speed, pos_scale, vel_scale,
                 n_samples=8, sample_steps=10, conflict_radius=10.0, mode="diffusion",
                 probe_accels=None, history_len=5, future_len=20, n_neighbors=5,
                 oracle_fn=None, guidance=1.0, deterministic_sampling=True,
                 latent_seed=0, conflict_point=None,
                 approach_edge_prefixes=(), decision_distance=None):
        self.model = world_model
        self.device = device
        self.dt = dt
        self.max_speed = max_speed
        self.pos_scale = pos_scale
        self.vel_scale = vel_scale
        self.n_samples = n_samples
        self.sample_steps = sample_steps
        self.conflict_radius = conflict_radius
        self.mode = mode
        self.oracle_fn = oracle_fn
        # A zero plan makes the guided and unguided logits identical, so the
        # history-only ablation is unaffected by this and stays a fair control.
        self.guidance = float(guidance)
        self.deterministic_sampling = bool(deterministic_sampling)
        self.latent_seed = int(latent_seed)
        self.conflict_point = (
            None if conflict_point is None
            else np.asarray(conflict_point, dtype=np.float32))
        self.approach_edge_prefixes = tuple(approach_edge_prefixes or ())
        self.decision_distance = (
            None if decision_distance is None else float(decision_distance))
        self.feature_blocks = {
            "risk", "clear", "spread", "intent", "shift", "waypoint"}
        self.probe_accels = tuple(
            DEFAULT_PROBE_ACCELS if probe_accels is None else probe_accels)

        if world_model is not None:
            self.history_len = world_model.history_len
            self.future_len = world_model.future_len
            self.n_neighbors = world_model.n_neighbors
        else:
            self.history_len = history_len
            self.future_len = future_len
            self.n_neighbors = n_neighbors
        # Per probe:
        #   0-3  risk, mean/min clearance, spread     (ego geometry vs forecast)
        #   4-5  P(yield), P(contest)                 (intention head / oracle)
        #   6    neighbour shift vs the hold probe    (influence; 0 if y does not
        #        move when u changes — geometry and history are identically 0)
        #   7-10 predicted closest-neighbour xy at mid and end of the horizon
        #        (a trajectory summary, not a scalar margin)
        # Contrasts: assert − yield on risk, clearance, spread, P(yield), shift.
        self.n_probe_features = 11
        self.n_contrast_features = 5
        self.feature_dim = (len(self.probe_accels) * self.n_probe_features
                            + self.n_contrast_features)

    def zero_features(self):
        return np.zeros(self.feature_dim, dtype=np.float32)

    def __call__(self, frames, ego_id, ego_speed):
        """Return the belief vector for the current step."""
        scene = extract_scene(
            frames, len(frames) - 1, ego_id,
            self.history_len, self.n_neighbors,
            conflict_point=self.conflict_point,
            approach_edge_prefixes=self.approach_edge_prefixes,
            decision_distance=self.decision_distance)
        if scene is None or not scene["neighbor_ids"]:
            return self.zero_features()
        yaw_rate = ego_yaw_rate(scene["history"], self.dt)
        decision_weights = self._decision_weights(frames, scene)
        if self.mode in ("geometry", "kernel", "oracle"):
            return self._geometry_features(
                scene, ego_speed, yaw_rate, decision_weights)

        history_raw = torch.from_numpy(scene["history"]).float().unsqueeze(0).to(self.device)
        history = history_raw.clone()
        history[..., :2] /= self.pos_scale
        history[..., 2:4] /= self.vel_scale

        n_probes = len(self.probe_accels)
        plans = []
        for accel in self.probe_accels:
            plan = synthetic_plan(ego_speed, accel, self.future_len, self.dt,
                                  self.max_speed, yaw_rate=yaw_rate)
            plan[..., :2] /= self.pos_scale
            plan[..., 2] /= self.vel_scale
            plans.append(plan)
        plan_batch = torch.from_numpy(np.stack(plans)).float().to(self.device)
        history_batch = history.expand(n_probes, -1, -1, -1)

        model_plan = torch.zeros_like(plan_batch) if self.mode == "history" else plan_batch
        # The batch dimension enumerates probes for one scene, so a shared latent
        # draw makes these true counterfactuals of each other: differences across
        # probes are the effect of u, not two independent sampling errors.
        generator = None
        eta = 1.0
        if self.deterministic_sampling:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.latent_seed)
            eta = 0.0
        samples = self.model.sample(
            history_batch, model_plan, n_samples=self.n_samples,
            steps=self.sample_steps, eta=eta, common_noise=True,
            generator=generator)
        samples = decode_samples(samples, history_raw.expand(n_probes, -1, -1, -1))
        if self.mode == "mean":
            samples = samples.mean(dim=1, keepdim=True)

        valid_mask = torch.from_numpy(
            (scene["history"][-1, 1:, 4] > 0).astype(np.float32)
        ).to(self.device)

        # Mean neighbour path per probe — the object whose *change across
        # probes* is influence, and whose *location* is a trajectory the
        # policy can actually see.
        pred_mean = samples.mean(dim=1)  # (n_probes, F, K, 2)
        hold_i = self._hold_index()
        hold_mean = pred_mean[hold_i]

        ego_paths, distances = [], []
        for accel in self.probe_accels:
            ego_plan_m = synthetic_plan(ego_speed, accel, self.future_len, self.dt,
                                        self.max_speed, yaw_rate=yaw_rate)
            ego_xy = torch.from_numpy(ego_plan_m[:, :2]).float().to(self.device)
            ego_paths.append(ego_xy)
            d = torch.linalg.norm(samples[len(distances)] - ego_xy[None, :, None, :],
                                  dim=-1)
            distances.append(d + (1.0 - valid_mask)[None, None, :] * 1e3)

        # Relevance is fixed at the hold probe. If the weights moved with u, the
        # assert-minus-yield contrast would blend the change in intentions with a
        # change in *which* neighbour is being summarised, and the second term is
        # larger than the first.
        weights = decision_weights
        if weights is not None:
            weights = torch.from_numpy(weights).to(self.device)
        else:
            weights = self._relevance(
                distances[hold_i].amin(dim=(0, 1)), valid_mask)
        cv_core = self._cv_core_stats(scene, ego_speed, yaw_rate)

        features = []
        probe_stats = []
        for p, accel in enumerate(self.probe_accels):
            distance = distances[p]
            per_sample_min = distance.min(dim=-1).values.min(dim=-1).values

            # All learned arms receive the exact same deterministic CV geometry
            # as the geometry baseline. The world model contributes only
            # response uncertainty, intention and trajectory residual features.
            risk, mean_clear, min_clear = cv_core[p]
            spread = float(per_sample_min.std(unbiased=False)) / 50.0

            intents = self.model.predict_intentions(
                history_batch[p:p + 1], model_plan[p:p + 1],
                guidance=self.guidance)
            p_yield, p_contest = self._pool_intents(intents[0], weights)

            shift = self._neighbor_shift(pred_mean[p], hold_mean, valid_mask)
            mid_xy, end_xy = self._closest_waypoints(pred_mean[p], weights)
            stats = (risk, mean_clear, min_clear, spread, p_yield, p_contest,
                     shift, mid_xy[0], mid_xy[1], end_xy[0], end_xy[1])
            features.extend(stats)
            probe_stats.append(stats)

        return self._mask_features(self._with_contrast(features, probe_stats))

    def _cv_core_stats(self, scene, ego_speed, yaw_rate):
        last = scene["history"][-1]
        neigh = last[1:]
        valid = neigh[:, 4]
        pred = np.zeros(
            (self.future_len, self.n_neighbors, 2), dtype=np.float32)
        for k in range(self.n_neighbors):
            if valid[k] < 0.5:
                pred[:, k] = 1e3
                continue
            x, y, vx, vy = neigh[k, :4]
            steps = np.arange(1, self.future_len + 1, dtype=np.float32)
            pred[:, k, 0] = x + vx * self.dt * steps
            pred[:, k, 1] = y + vy * self.dt * steps
        stats = []
        for accel in self.probe_accels:
            ego_xy = synthetic_plan(
                ego_speed, accel, self.future_len, self.dt, self.max_speed,
                yaw_rate=yaw_rate)[:, :2]
            distance = np.linalg.norm(pred - ego_xy[:, None], axis=-1)
            distance += (1.0 - valid)[None] * 1e3
            clear = float(distance.min()) / 50.0
            stats.append((
                float(distance.min() < self.conflict_radius), clear, clear))
        return stats

    def _relevance(self, closest_per_neighbor, valid_mask):
        """Softmax over predicted closest approach; ``None`` when nobody is valid."""
        if float(valid_mask.sum()) <= 0:
            return None
        logits = -closest_per_neighbor / RELEVANCE_TAU
        logits = logits.masked_fill(valid_mask < 0.5, -1e9)
        return torch.softmax(logits, dim=0)

    def _decision_weights(self, frames, scene):
        """Weight neighbours able to reach the conflict within the query horizon."""
        if self.conflict_point is None or not frames:
            return None
        vehicles = frames[-1]["vehicles"]
        horizon = self.future_len * self.dt
        ttc = np.full(self.n_neighbors, np.inf, dtype=np.float32)
        valid = np.zeros(self.n_neighbors, dtype=bool)
        for k, veh_id in enumerate(scene["neighbor_ids"]):
            state = vehicles.get(veh_id)
            if state is None:
                continue
            x, y, _, _, speed, heading = state[:6]
            approach = self.conflict_point - np.asarray([x, y])
            if float(np.dot(
                    approach, [np.cos(heading), np.sin(heading)])) < 0:
                continue
            ttc[k] = float(np.linalg.norm(approach)) / max(float(speed), 0.5)
            valid[k] = ttc[k] <= horizon + self.dt
        if not valid.any():
            return None
        logits = np.where(valid, -ttc / max(horizon, 1e-6), -1e9)
        weights = np.exp(logits - logits.max())
        return (weights / weights.sum()).astype(np.float32)

    def _geometry_features(self, scene, ego_speed, yaw_rate=0.0,
                           decision_weights=None):
        """CV neighbour roll-out vs each probe ego plan. No learned model.

        Neighbour predictions do not depend on the probe, so the influence
        (shift) feature is identically zero. Waypoints still show where the
        constant-velocity forecast puts the closest car — trajectory, not just
        a scalar margin — which is the most a geometry policy is allowed to see.
        """
        last = scene["history"][-1]
        neigh = last[1:]
        valid = neigh[:, 4]
        pred = np.zeros((self.future_len, self.n_neighbors, 2), dtype=np.float32)
        for k in range(self.n_neighbors):
            if valid[k] < 0.5:
                pred[:, k] = 1e3
                continue
            x, y, vx, vy = neigh[k, :4]
            for f in range(self.future_len):
                pred[f, k, 0] = x + vx * self.dt * (f + 1)
                pred[f, k, 1] = y + vy * self.dt * (f + 1)

        dists = []
        for accel in self.probe_accels:
            ego_xy = synthetic_plan(
                ego_speed, accel, self.future_len, self.dt, self.max_speed,
                yaw_rate=yaw_rate)[:, :2]
            d = np.linalg.norm(pred - ego_xy[:, None, :], axis=-1)
            dists.append(d + (1.0 - valid)[None, :] * 1e3)

        # Relevance fixed at the hold probe, matching the learned encoder, so an
        # oracle contrast reports the change in intentions and nothing else.
        hold = dists[self._hold_index()].min(axis=0)
        weights = decision_weights
        if weights is None and float(valid.sum()) > 0:
            logits = np.where(valid >= 0.5, -hold / RELEVANCE_TAU, -1e9)
            weights = np.exp(logits - logits.max())
            weights = weights / weights.sum()
        mid_xy, end_xy = self._closest_waypoints_np(pred, weights)

        features, probe_stats = [], []
        for accel, dist in zip(self.probe_accels, dists):
            per_sample_min = float(dist.min())
            risk = float(per_sample_min < self.conflict_radius)
            clear = per_sample_min / 50.0
            p_yield, p_contest = 0.0, 0.0
            if self.mode in ("kernel", "oracle") and self.oracle_fn is not None:
                p_yield, p_contest = self._pool_oracle(
                    scene["neighbor_ids"], accel, weights)
            # Same neighbour forecast for every probe → influence is zero.
            stats = (risk, clear, clear, 0.0, p_yield, p_contest,
                     0.0, mid_xy[0], mid_xy[1], end_xy[0], end_xy[1])
            features.extend(stats)
            probe_stats.append(stats)
        return self._mask_features(self._with_contrast(features, probe_stats))

    def set_feature_blocks(self, blocks):
        known = {"risk", "clear", "spread", "intent", "shift", "waypoint"}
        blocks = {str(block).strip() for block in blocks if str(block).strip()}
        unknown = blocks - known
        if unknown:
            raise ValueError(f"unknown belief feature blocks: {sorted(unknown)}")
        self.feature_blocks = blocks
        return self

    def _mask_features(self, vector):
        vector = np.asarray(vector, dtype=np.float32).copy()
        block_offsets = {
            "risk": (0,), "clear": (1, 2), "spread": (3,),
            "intent": (4, 5), "shift": (6,),
            "waypoint": (7, 8, 9, 10),
        }
        for block, offsets in block_offsets.items():
            if block in self.feature_blocks:
                continue
            for probe in range(len(self.probe_accels)):
                base = probe * self.n_probe_features
                vector[[base + offset for offset in offsets]] = 0.0
        contrast_base = len(self.probe_accels) * self.n_probe_features
        contrast_blocks = ("risk", "clear", "spread", "intent", "shift")
        for i, block in enumerate(contrast_blocks):
            if block not in self.feature_blocks:
                vector[contrast_base + i] = 0.0
        return vector

    def _hold_index(self):
        accels = [float(a) for a in self.probe_accels]
        if 0.0 in accels:
            return accels.index(0.0)
        return 0

    def _neighbor_shift(self, pred_mean, hold_mean, valid_mask):
        """Mean displacement of predicted neighbours when the probe changes.

        This is the quantity geometry cannot fake: constant-velocity (and
        history-only) forecasts do not move with u, so the value is 0.
        """
        if float(valid_mask.sum()) <= 0:
            return 0.0
        delta = torch.linalg.norm(pred_mean - hold_mean, dim=-1)  # (F, K)
        per_k = delta.mean(dim=0)
        # A five-metre response is order one; dividing by the generic 50 m
        # position scale made this causal feature numerically disappear.
        return float((per_k * valid_mask).sum() / valid_mask.sum()) / 5.0

    def _closest_waypoints(self, pred_mean, weights):
        """Mid- and end-horizon xy of the most relevant predicted neighbour."""
        mid = self.future_len // 2
        end = self.future_len - 1
        if weights is None:
            return (0.0, 0.0), (0.0, 0.0)
        k = int(torch.argmax(weights).item())
        mid_xy = pred_mean[mid, k] / 50.0
        end_xy = pred_mean[end, k] / 50.0
        return (float(mid_xy[0]), float(mid_xy[1])), (float(end_xy[0]), float(end_xy[1]))

    @staticmethod
    def _closest_waypoints_np(pred, weights):
        if weights is None:
            return (0.0, 0.0), (0.0, 0.0)
        k = int(weights.argmax())
        mid = pred.shape[0] // 2
        end = pred.shape[0] - 1
        return tuple(float(x) / 50.0 for x in pred[mid, k]), tuple(
            float(x) / 50.0 for x in pred[end, k])

    def _pool_intents(self, intents, weights):
        """Report the most dangerous partner, not the mean across K slots.

        Averaging a yielder with a contester reads as 0.5 and hides the stream
        that will actually collide. Cross has two independent approaches; RA
        used to mix the wrong ring arc into the same pool.
        """
        if weights is None:
            return 0.0, 0.0
        if isinstance(intents, torch.Tensor):
            w = weights if isinstance(weights, torch.Tensor) else torch.as_tensor(
                weights, device=intents.device, dtype=intents.dtype)
            n = min(intents.shape[0], w.shape[0])
            mask = w[:n] > 1e-6
            if float(mask.sum()) <= 0:
                return 0.0, 0.0
            return (float(intents[:n][mask, 0].min()),
                    float(intents[:n][mask, 1].max()))
        intents = np.asarray(intents)
        w = np.asarray(weights)
        n = min(len(intents), len(w))
        mask = w[:n] > 1e-6
        if not np.any(mask):
            return 0.0, 0.0
        return (float(intents[:n][mask, 0].min()),
                float(intents[:n][mask, 1].max()))

    def _pool_oracle(self, neighbor_ids, accel, weights):
        """Worst-partner true intent, matching the learned pooling exactly."""
        intents = self.oracle_fn(neighbor_ids, float(accel))
        if intents is None or len(intents) == 0:
            return 0.0, 0.0
        return self._pool_intents(intents, weights)

    @staticmethod
    def _with_contrast(features, probe_stats):
        yld, ast = probe_stats[0], probe_stats[-1]
        features.extend([
            ast[0] - yld[0],   # risk
            ast[1] - yld[1],   # clearance
            ast[3] - yld[3],   # spread
            ast[4] - yld[4],   # P(yield)
            ast[6] - yld[6],   # neighbour shift (influence)
        ])
        return np.nan_to_num(np.asarray(features, dtype=np.float32))
