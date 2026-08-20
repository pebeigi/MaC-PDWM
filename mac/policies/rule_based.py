"""Rule-based ego policies used as baselines and as data-collection behaviours."""
import numpy as np


CLEARANCE = 12.0     # metres of conflict zone the vehicle must clear
COMMIT_DISTANCE = 30.0  # inside this range an accepted gap becomes a commitment
STOP_MARGIN = 5.0    # hold this far short of the conflict point while yielding
CREEP_SPEED = 2.0


class TimeGapPolicy:
    """Gap-acceptance controller for the conflict point.

    Occupancy of the conflict zone is predicted as a time interval for the ego
    and for each approaching neighbour; the ego yields when those intervals
    overlap. This mirrors how a human driver reads a crossing, and the
    ``aggressiveness``/``noise`` knobs give the behavioural diversity the world
    model is trained on.
    """

    def __init__(self, env, accept_gap=1.5, aggressiveness=0.0, noise=0.0, rng=None):
        self.env = env
        self.accept_gap = accept_gap
        self.aggressiveness = aggressiveness
        self.noise = noise
        self.rng = rng or np.random.default_rng()
        self.actions = np.asarray(env.cfg.discrete_actions)
        self.committed = {}

    def reset(self):
        self.committed = {}

    def _decode(self, obs_row):
        max_speed = self.env.spec.max_speed
        speed = obs_row[0] * max_speed
        d_conflict = obs_row[2] * 100.0
        n_dim = self.env.neighbor_feature_dim
        neighbors = []
        for slot in range(self.env.cfg.n_neighbors):
            start = self.env.ego_feature_dim + slot * n_dim
            block = obs_row[start:start + n_dim]
            if block[-1] < 0.5:
                continue
            neighbors.append({
                "speed": block[4] * max_speed,
                "d_conflict": block[5] * 100.0,
            })
        return speed, d_conflict, neighbors

    def _blocked(self, speed, d_conflict, neighbors):
        margin = max(0.2, self.accept_gap - self.aggressiveness)
        # Judging the crossing at a reference speed rather than the current one
        # avoids the deadlock where slowing down always looks safe.
        v_ref = max(speed, 5.0)
        ego_enter = d_conflict / v_ref
        ego_exit = (d_conflict + CLEARANCE) / v_ref

        for neighbor in neighbors:
            d_other = neighbor["d_conflict"]
            if d_other <= 0.0 or d_other > 90.0:
                continue
            v_other = max(neighbor["speed"], 0.5)
            other_enter = d_other / v_other
            other_exit = (d_other + CLEARANCE) / v_other
            if ego_enter < other_exit + margin and other_enter < ego_exit + margin:
                return True
        return False

    def act(self, obs):
        obs = np.atleast_2d(obs)
        dt = self.env.dt
        chosen = []
        for i, row in enumerate(obs):
            speed, d_conflict, neighbors = self._decode(row)

            if d_conflict <= 0.0 or self.committed.get(i, False):
                desired = self.env.cfg.max_accel
            elif self._blocked(speed, d_conflict, neighbors):
                self.committed[i] = False
                # Creep up to the stop line and hold there. The slow roll is
                # itself the intent signal; entering the zone is not.
                room = max(d_conflict - STOP_MARGIN, 0.0)
                target = min(CREEP_SPEED, np.sqrt(2 * 1.5 * room))
                desired = float(np.clip((target - speed) / dt, -self.env.cfg.max_decel, 1.5))
            else:
                # Only latch a commitment once the gap has been judged safe.
                self.committed[i] = d_conflict < COMMIT_DISTANCE
                desired = 2.0

            if self.noise > 0 and self.rng.random() < self.noise:
                idx = int(self.rng.integers(0, len(self.actions)))
            else:
                idx = int(np.argmin(np.abs(self.actions - desired)))
            chosen.append(idx)
        return np.array(chosen, dtype=np.int64)


class ConstantSpeedPolicy:
    """Always-go baseline; useful to measure how unsafe an uncooperative ego is."""

    def __init__(self, env):
        self.env = env
        self.idx = int(np.argmin(np.abs(np.asarray(env.cfg.discrete_actions) - 1.5)))

    def act(self, obs):
        return np.full(len(np.atleast_2d(obs)), self.idx, dtype=np.int64)


class ConstantAccelPolicy:
    """Holds one acceleration for the whole episode.

    The belief encoder probes the world model with constant-acceleration plans,
    so the training data has to contain episodes in which the ego actually drives
    that way; otherwise every counterfactual query is off-distribution.
    """

    def __init__(self, env, accel):
        self.env = env
        self.accel = float(accel)
        self.idx = int(np.argmin(np.abs(np.asarray(env.cfg.discrete_actions) - accel)))

    def act(self, obs):
        return np.full(len(np.atleast_2d(obs)), self.idx, dtype=np.int64)


class OpenLoopPlanPolicy:
    """Execute a pre-sampled acceleration sequence.

    Randomising full $F$-step (and multi-segment) plans, not only one-step
    actions, is what identifies $p(y\\mid h,\\mathrm{do}(u_{1:F}))$ on the
    probe family the planner queries.
    """

    def __init__(self, env, accels):
        self.env = env
        self.accels = [float(a) for a in accels]
        self.actions = np.asarray(env.cfg.discrete_actions)
        self.t = 0

    def reset(self):
        self.t = 0

    def act(self, obs):
        a = self.accels[min(self.t, len(self.accels) - 1)]
        self.t += 1
        idx = int(np.argmin(np.abs(self.actions - a)))
        return np.full(len(np.atleast_2d(obs)), idx, dtype=np.int64)
