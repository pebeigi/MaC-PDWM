"""Vehicle-level planning environment on top of SUMO.

The ego vehicles are driven directly through TraCI with SUMO's own safety
checks disabled, so right-of-way at the conflict point has to be resolved by
the policy rather than by the simulator. Background traffic keeps its default
car-following model and therefore reacts to the ego's motion, which is what
makes motion usable as a communication channel.
"""
import copy
import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import traci

from mac.data.scene import rank_neighbor_ids
from mac.envs.social_drivers import SocialDriverManager

SCENARIO_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scenarios")


def parse_type_probs(text):
    """Parse ``yielder,contester,reactive`` weights from a CLI string."""
    parts = [float(p.strip()) for p in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected three type probabilities, got {text!r}")
    return tuple(parts)

# Bitmask 0 disables every SUMO safety check (safe speed, right of way,
# junction blocking) for the controlled vehicles.
EGO_SPEED_MODE = 0
EGO_LANE_CHANGE_MODE = 0


@dataclass
class ScenarioSpec:
    name: str
    sumocfg: str
    net_file: str
    ego_routes: List[str]
    background_routes: Dict[str, float]  # route id -> insertion rate (veh/s)
    conflict_point: Tuple[float, float]
    max_speed: float
    ego_depart_pos: float = 20.0
    ego_depart_speed: float = 8.0
    # Euclidean decision radius. Roundabout needs a tighter ball: the conflict
    # sits on a 20 m ring, so D=35 covers half the circle and drivers commit on
    # the wrong arc.
    decision_distance: float = 35.0
    # Optional edge-id prefixes that may commit. Empty = any approaching edge.
    # Used on the roundabout to keep commits on the south-merge feeders.
    approach_edge_prefixes: Tuple[str, ...] = ()
    # Kinematic yield target as a fraction of current speed. None = default
    # 0.3. Roundabout uses a deeper brake because contest cannot surge on the
    # speed-limited ring; yield is the only identifiable channel.
    concession_speed_fraction: Optional[float] = None


SCENARIOS: Dict[str, ScenarioSpec] = {
    "cross": ScenarioSpec(
        name="cross",
        sumocfg=os.path.join(SCENARIO_ROOT, "unsignalized_cross", "cross.sumocfg"),
        net_file=os.path.join(SCENARIO_ROOT, "unsignalized_cross", "cross.net.xml"),
        ego_routes=["SN"],
        background_routes={"WE": 0.25, "EW": 0.25},
        conflict_point=(0.0, 0.0),
        max_speed=13.89,
        decision_distance=35.0,
    ),
    "merge": ScenarioSpec(
        name="merge",
        sumocfg=os.path.join(SCENARIO_ROOT, "merge", "merge.sumocfg"),
        net_file=os.path.join(SCENARIO_ROOT, "merge", "merge.net.xml"),
        ego_routes=["RAMP"],
        background_routes={"MAIN": 0.30},
        conflict_point=(0.0, 0.0),
        max_speed=22.22,
        ego_depart_pos=10.0,
        ego_depart_speed=12.0,
        decision_distance=35.0,
    ),
    # South entry merge of a 20 m single-lane roundabout. Background OD is
    # balanced across the four arms (through + turns); ego (SN) claims a gap
    # at the south merge against traffic that actually uses circ_WS.
    "roundabout": ScenarioSpec(
        name="roundabout",
        sumocfg=os.path.join(SCENARIO_ROOT, "roundabout", "roundabout.sumocfg"),
        net_file=os.path.join(SCENARIO_ROOT, "roundabout", "roundabout.net.xml"),
        ego_routes=["SN"],
        # Rates ≈ veh/s. Total ~0.50 ≈ old WE+NS demand, but origins/destinations
        # are spread so N/W→S no longer dominates the ring.
        # Routes that pass the south merge (circ_WS): WE, NS, WS, ES, NE, WN.
        background_routes={
            "WE": 0.07, "EW": 0.07, "NS": 0.05,          # through
            "WN": 0.05, "ES": 0.045, "NE": 0.045,        # left
            "WS": 0.03, "EN": 0.045, "NW": 0.04,         # right (non-S)
            "SE": 0.025, "SW": 0.025,                    # light S-origin turns
        },
        conflict_point=(0.0, -20.0),
        max_speed=13.89,
        ego_depart_pos=20.0,
        ego_depart_speed=8.0,
        decision_distance=22.0,
        # West approach into the south merge only. circ_NW is Euclidean-near
        # but path-far; allowing it made drivers commit ~100 m from the ego.
        approach_edge_prefixes=("W_in", ":nW_", "circ_WS", ":nS_"),
        concession_speed_fraction=0.15,
    ),
}


@dataclass
class EnvConfig:
    scenario: str = "cross"
    n_ego: int = 1
    sim_step: float = 0.2
    action_repeat: int = 2          # decision interval = sim_step * action_repeat
    horizon: int = 150              # decision steps per episode
    history_len: int = 5
    n_neighbors: int = 5
    neighbor_radius: float = 80.0
    max_accel: float = 3.0
    max_decel: float = 6.0
    discrete_actions: Optional[List[float]] = None
    warmup_steps: int = 50          # sim steps of background traffic before ego insertion
    bg_rate_scale: float = 1.0      # scales the scenario's background insertion rates
    gui: bool = False
    seed: int = 0
    collision_penalty: float = 20.0
    success_reward: float = 20.0
    # Without an explicit cost for never crossing, creeping short of the conflict
    # point is a stable local optimum worth more than any attempt to cross, and
    # every policy collapses onto it.
    timeout_penalty: float = 15.0
    # Waiting has to cost enough that hanging back for a natural gap is not a
    # free substitute for negotiating: at 0.02 a fifty-step delay costs 1.0
    # against a success bonus of 20, so a purely conservative policy is optimal
    # and no amount of belief quality can pay for itself.
    time_penalty: float = 0.06
    progress_weight: float = 0.2
    accel_weight: float = 0.005
    jerk_weight: float = 0.01
    proximity_weight: float = 0.05
    proximity_threshold: float = 8.0
    courtesy_weight: float = 0.15   # penalty per hard brake the ego induces in others
    courtesy_grace_steps: int = 3   # steps after a concession that are not charged
    type_probs: Tuple[float, float, float] = (0.1, 0.1, 0.8)
    # Strength of the influence channel; 0.0 makes latent types independent of
    # the ego's motion (the no-channel control condition).
    beta_intent: float = 2.5
    # Weight on the priority (TTC) term. Together with beta_intent this sets how
    # much of the decision is geometry and how much is communication.
    beta_margin: float = 0.3
    # Distance at which a reactive driver commits. None = use ScenarioSpec.
    decision_distance: Optional[float] = None
    intent_window: float = 0.8

    def __post_init__(self):
        probs = np.asarray(self.type_probs, dtype=float)
        if probs.shape != (3,) or (probs < 0).any() or probs.sum() <= 0:
            raise ValueError("type_probs must be three non-negative weights "
                             "(yielder, contester, reactive)")
        self.type_probs = tuple(float(x) for x in (probs / probs.sum()))
        if self.scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {self.scenario!r}")
        if self.decision_distance is None:
            self.decision_distance = float(SCENARIOS[self.scenario].decision_distance)
        if self.discrete_actions is None:
            # The mean of this set is zero. An action set skewed towards braking
            # (the natural choice, since emergency decelerations are larger than
            # any comfortable acceleration) makes a maximum-entropy policy drift
            # to a standstill, so the agent never reaches the goal during
            # exploration and never observes the arrival reward at all.
            self.discrete_actions = [-4.0, -2.0, 0.0, 1.0, 2.0, 3.0]


@dataclass
class EgoState:
    veh_id: str
    route: str
    active: bool = True
    done: bool = False
    outcome: str = ""
    last_accel: float = 0.0
    prev_accel: float = 0.0
    history: List[np.ndarray] = field(default_factory=list)
    collision_partner_type: str = ""
    collision_partner_resolved: str = ""


class SumoPlanningEnv:
    """Multi-ego planning env with a gym-like API.

    ``step`` accepts one action per ego (index into ``cfg.discrete_actions`` or a
    raw acceleration when ``continuous=True``) and returns stacked observations,
    per-ego rewards, per-ego done flags and an info dict.
    """

    def __init__(self, cfg: Optional[EnvConfig] = None, label: Optional[str] = None, continuous: bool = False):
        self.cfg = cfg or EnvConfig()
        self.spec = SCENARIOS[self.cfg.scenario]
        self.continuous = continuous
        self.label = label or f"mac_{os.getpid()}_{id(self)}"
        self.conn = None
        self.rng = np.random.default_rng(self.cfg.seed)
        self.dt = self.cfg.sim_step * self.cfg.action_repeat

        self.egos: List[EgoState] = []
        self.step_count = 0
        self.episode_id = 0
        self._episode_seed = int(self.cfg.seed)
        self._sim_time = 0.0
        self._bg_counter = 0
        self._pending_bg: Dict[str, float] = {}
        self._frames: List[dict] = []
        self._last_snapshot: Dict[str, dict] = {}
        concession_frac = self.spec.concession_speed_fraction
        driver_kwargs = dict(
            type_probs=self.cfg.type_probs,
            beta_intent=self.cfg.beta_intent, beta_margin=self.cfg.beta_margin,
            decision_distance=self.cfg.decision_distance,
            intent_window=self.cfg.intent_window,
            courtesy_grace_steps=self.cfg.courtesy_grace_steps,
            approach_edge_prefixes=self.spec.approach_edge_prefixes,
            scenario=self.spec.name,
        )
        if concession_frac is not None:
            driver_kwargs["concession_speed_fraction"] = concession_frac
        self.drivers = SocialDriverManager(
            self.spec.conflict_point, self.rng, **driver_kwargs,
        )

        self.ego_feature_dim = 6
        self.neighbor_feature_dim = 7
        self.obs_dim = self.ego_feature_dim + self.cfg.n_neighbors * self.neighbor_feature_dim
        self.action_dim = 1 if continuous else len(self.cfg.discrete_actions)

    # ------------------------------------------------------------------ setup

    def _sumo_binary(self):
        binary = "sumo-gui" if self.cfg.gui else "sumo"
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            candidate = os.path.join(sumo_home, "bin", binary)
            if os.path.exists(candidate):
                return candidate
        return binary

    def _start(self, seed: int):
        cmd = [
            self._sumo_binary(),
            "-c", self.spec.sumocfg,
            "--step-length", str(self.cfg.sim_step),
            "--seed", str(seed),
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--time-to-teleport", "-1",
            # 'warn' keeps the vehicles in place but still registers the event;
            # 'none' would disable detection altogether.
            "--collision.action", "warn",
            "--collision.check-junctions", "true",
            "--collision.mingap-factor", "0",
            # Saved states are rounded to two decimals by default. That drift is
            # enough to move a commitment by one step, which reorders the draws
            # taken from the shared generator and makes two branches of the
            # "same" scene diverge outright.
            "--save-state.precision", "12",
            # The simulator's own generators drive car-following noise and
            # insertion; without them in the state, restored branches diverge a
            # few steps in even at full precision.
            "--save-state.rng", "true",
        ]
        traci.start(cmd, label=self.label)
        self.conn = traci.getConnection(self.label)

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def save_branch_state(self, path):
        """Save SUMO and Python-side state for matched counterfactual branches."""
        if self.conn is None:
            raise RuntimeError("environment is not running")
        self.conn.simulation.saveState(path)
        driver_state = {
            key: copy.deepcopy(value)
            for key, value in self.drivers.__dict__.items()
            if key not in ("rng", "conflict_point")
        }
        return {
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "step_count": self.step_count,
            "episode_id": self.episode_id,
            "sim_time": self._sim_time,
            "bg_counter": self._bg_counter,
            "pending_bg": copy.deepcopy(self._pending_bg),
            "frames": copy.deepcopy(self._frames),
            "last_snapshot": copy.deepcopy(self._last_snapshot),
            "egos": copy.deepcopy(self.egos),
            "induced_brakes": self._induced_brakes,
            "drivers": driver_state,
        }

    def load_branch_state(self, path, state, fresh=True):
        """Restore a state produced by :meth:`save_branch_state`.

        Branches are loaded into a freshly started simulator by default. Rewinding
        a simulator that has already run a branch does not undo its record of
        which vehicles have arrived, so a vehicle that completed its route in an
        earlier branch is never re-inserted: it stays pinned at its departure
        position for the rest of the episode while simulation time advances
        normally. Nothing raises, and the resulting comparison silently measures
        a frozen ego instead of a response to the plan.
        """
        if self.conn is None and not fresh:
            raise RuntimeError("environment is not running")
        if fresh:
            self.close()
            self._start(self._episode_seed)
        self.conn.simulation.loadState(path)
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.step_count = state["step_count"]
        self.episode_id = state["episode_id"]
        self._sim_time = state["sim_time"]
        self._bg_counter = state["bg_counter"]
        self._pending_bg = copy.deepcopy(state["pending_bg"])
        self._frames = copy.deepcopy(state["frames"])
        self._last_snapshot = copy.deepcopy(state["last_snapshot"])
        self.egos = copy.deepcopy(state["egos"])
        self._induced_brakes = state["induced_brakes"]
        for key, value in state["drivers"].items():
            setattr(self.drivers, key, copy.deepcopy(value))
        # A restored simulator state does not carry in-flight speed commands, so
        # a branch loaded mid-concession would diverge from the saved rollout.
        self.drivers.reapply_influences(self.conn, self.dt)
        self._verify_branch_state(state)

    def _verify_branch_state(self, state, tol=1e-3):
        """Fail loudly when a restored branch does not match the saved scene."""
        present = set(self.conn.vehicle.getIDList())
        for ego in self.egos:
            if ego.done or ego.veh_id not in state["last_snapshot"]:
                continue
            if ego.veh_id not in present:
                raise RuntimeError(
                    f"branch restore lost ego {ego.veh_id}: matched-branch "
                    "comparisons from this state would be meaningless")
            saved = state["last_snapshot"][ego.veh_id]
            x, y = self.conn.vehicle.getPosition(ego.veh_id)
            if abs(x - saved["x"]) > tol or abs(y - saved["y"]) > tol:
                raise RuntimeError(
                    f"branch restore moved ego {ego.veh_id} from "
                    f"({saved['x']:.4f}, {saved['y']:.4f}) to ({x:.4f}, {y:.4f})")

    # ------------------------------------------------------------------ reset

    def reset(self, seed: Optional[int] = None):
        self.close()
        if seed is None:
            episode_seed = int(self.rng.integers(0, 2**31 - 1))
        else:
            # SUMO's --seed only controls the simulator's own randomness. Warm-up
            # length, background headways, ego insertion, driver types and the
            # channel coin flips all come from self.rng, so an explicit seed has
            # to reset that too or the "same" evaluation episode differs between
            # arms whose training rollouts consumed different numbers of draws.
            # Reseeding the bit generator in place keeps SocialDriverManager's
            # reference to this generator valid.
            episode_seed = int(seed)
            self.rng.bit_generator.state = (
                np.random.default_rng(episode_seed).bit_generator.state)
        self._episode_seed = episode_seed
        self._start(episode_seed)

        self.step_count = 0
        self.episode_id += 1
        self._bg_counter = 0
        self._sim_time = 0.0
        self._pending_bg = {
            route: float(self.rng.exponential(1.0))
            for route in self.spec.background_routes
        }
        self._frames = []
        self.egos = []
        self._induced_brakes = 0
        self.drivers.reset()

        # A random warm-up shifts the phase between the ego and the oncoming
        # stream, so the conflict geometry differs from episode to episode.
        warmup = int(self.rng.integers(self.cfg.warmup_steps, self.cfg.warmup_steps + 60))
        for _ in range(warmup):
            self._spawn_background()
            self.conn.simulationStep()
            self.drivers.apply_pending(self.conn)
            sub_snapshot = self._vehicle_snapshot()
            self.drivers.track_braking(
                self.conn, sub_snapshot,
                self._ego_states_from_snapshot(sub_snapshot),
                self.cfg.sim_step)

        self._insert_egos()
        # One step so the inserted egos are actually present in the network.
        self.conn.simulationStep()
        for ego in self.egos:
            if ego.veh_id in self.conn.vehicle.getIDList():
                self.conn.vehicle.setSpeedMode(ego.veh_id, EGO_SPEED_MODE)
                self.conn.vehicle.setLaneChangeMode(ego.veh_id, EGO_LANE_CHANGE_MODE)

        obs = self._build_observations()
        self._record_frame()
        return obs

    def _insert_egos(self):
        for i in range(self.cfg.n_ego):
            route = self.spec.ego_routes[i % len(self.spec.ego_routes)]
            veh_id = f"ego_{self.episode_id}_{i}"
            depart_pos = self.spec.ego_depart_pos + i * 15.0 + float(self.rng.uniform(0.0, 15.0))
            depart_speed = self.spec.ego_depart_speed * float(self.rng.uniform(0.8, 1.15))
            self.conn.vehicle.add(
                vehID=veh_id,
                routeID=route,
                typeID="ego",
                depart="now",
                departLane="first",
                departPos=str(depart_pos),
                departSpeed=str(depart_speed),
            )
            self.egos.append(EgoState(veh_id=veh_id, route=route))

    def _spawn_background(self):
        """Insert background traffic with exponential headways.

        Deterministic headways would lock the ego into the same conflict
        geometry every episode, so arrival times are sampled from a Poisson
        process instead.
        """
        self._sim_time += self.cfg.sim_step
        for route, rate in self.spec.background_routes.items():
            rate *= self.cfg.bg_rate_scale
            if rate <= 0:
                continue
            while self._pending_bg[route] <= self._sim_time:
                self._pending_bg[route] += float(self.rng.exponential(1.0 / rate))
                veh_id = f"bg_{self.episode_id}_{self._bg_counter}"
                self._bg_counter += 1
                try:
                    self.conn.vehicle.add(
                        vehID=veh_id,
                        routeID=route,
                        typeID="background",
                        depart="now",
                        departLane="first",
                        departPos="base",
                        departSpeed="max",
                    )
                    self.drivers.register(veh_id)
                except traci.TraCIException:
                    # Insertion can fail when the entry lane is occupied; the
                    # vehicle is simply skipped for this episode.
                    pass

    # ------------------------------------------------------------------- step

    def step(self, actions):
        actions = np.atleast_1d(actions)
        accels = []
        for i, ego in enumerate(self.egos):
            if self.continuous:
                accel = float(np.clip(actions[i], -self.cfg.max_decel, self.cfg.max_accel))
            else:
                accel = float(self.cfg.discrete_actions[int(actions[i])])
            accels.append(accel)

        alive_ids = set(self.conn.vehicle.getIDList())
        for ego, accel in zip(self.egos, accels):
            if not ego.active or ego.veh_id not in alive_ids:
                continue
            speed = self.conn.vehicle.getSpeed(ego.veh_id)
            target = float(np.clip(speed + accel * self.dt, 0.0, self.spec.max_speed))
            self.conn.vehicle.setSpeed(ego.veh_id, target)
            ego.prev_accel = ego.last_accel
            ego.last_accel = accel

        collisions: Dict[str, bool] = {}
        arrived: Dict[str, bool] = {}
        brakes_before = self.drivers.ego_induced_brakes
        for _ in range(self.cfg.action_repeat):
            self._spawn_background()
            self.conn.simulationStep()
            self.drivers.apply_pending(self.conn)
            for collider in self.conn.simulation.getCollidingVehiclesIDList():
                collisions[collider] = True
            for veh_id in self.conn.simulation.getArrivedIDList():
                arrived[veh_id] = True

        self.step_count += 1
        obs = self._build_observations()
        self.drivers.update(self.conn, self._last_snapshot, self._ego_states(), self.dt)
        self._induced_brakes = self.drivers.ego_induced_brakes - brakes_before
        rewards, dones, infos = self._evaluate(collisions, arrived)
        self._record_frame()

        timeout = self.step_count >= self.cfg.horizon
        if timeout:
            for i, ego in enumerate(self.egos):
                if not ego.done:
                    ego.done = True
                    ego.outcome = ego.outcome or "timeout"
                    rewards[i] -= self.cfg.timeout_penalty
                    dones[i] = True

        info = {
            "outcomes": [ego.outcome for ego in self.egos],
            "timeout": timeout,
            "step": self.step_count,
            "induced_brakes": self._induced_brakes,
            "driver_stats": self.drivers.stats(),
            "collision_partner_type": [
                ego.collision_partner_type for ego in self.egos],
            "collision_partner_resolved": [
                ego.collision_partner_resolved for ego in self.egos],
            **infos,
        }
        return obs, rewards, dones, info

    def _ego_states(self):
        return self._ego_states_from_snapshot(self._last_snapshot)

    def _ego_states_from_snapshot(self, snapshot):
        states = []
        for ego in self.egos:
            state = snapshot.get(ego.veh_id)
            if state is None or ego.done:
                continue
            states.append({
                "speed": state["speed"],
                "accel": ego.last_accel,
                "d_conflict": self._signed_conflict_distance(state),
            })
        return states

    def _evaluate(self, collisions, arrived):
        rewards = np.zeros(len(self.egos), dtype=np.float32)
        dones = np.zeros(len(self.egos), dtype=bool)
        alive_ids = set(self.conn.vehicle.getIDList())
        min_gaps = []

        for i, ego in enumerate(self.egos):
            if ego.done:
                dones[i] = True
                continue

            if ego.veh_id in collisions:
                rewards[i] -= self.cfg.collision_penalty
                ego.done, ego.active, ego.outcome = True, False, "collision"
                dones[i] = True
                min_gaps.append(0.0)
                partners = [vid for vid in collisions if vid != ego.veh_id]
                if not partners and getattr(self, "_last_snapshot", None):
                    ego_state = self._last_snapshot.get(ego.veh_id)
                    if ego_state is not None:
                        others = []
                        for vid, state in self._last_snapshot.items():
                            if vid == ego.veh_id:
                                continue
                            others.append((
                                float(np.hypot(state["x"] - ego_state["x"],
                                               state["y"] - ego_state["y"])),
                                vid))
                        others.sort()
                        partners = [others[0][1]] if others else []
                partner = partners[0] if partners else None
                ego.collision_partner_type = (
                    self.drivers.types.get(partner, "unknown") if partner else "unknown")
                ego.collision_partner_resolved = (
                    self.drivers.resolved.get(partner, "unresolved") if partner else "unresolved")
                continue

            if ego.veh_id in arrived:
                rewards[i] += self.cfg.success_reward
                ego.done, ego.active, ego.outcome = True, False, "success"
                dones[i] = True
                continue

            if ego.veh_id not in alive_ids:
                # Removed without completing the route (teleport, collision
                # cleanup in an earlier sub-step). Not a success.
                ego.done, ego.active, ego.outcome = True, False, "lost"
                dones[i] = True
                continue

            speed = self.conn.vehicle.getSpeed(ego.veh_id)
            rewards[i] += self.cfg.progress_weight * (speed / self.spec.max_speed)
            rewards[i] -= self.cfg.time_penalty
            rewards[i] -= self.cfg.accel_weight * ego.last_accel**2
            jerk = (ego.last_accel - ego.prev_accel) / self.dt
            rewards[i] -= self.cfg.jerk_weight * (jerk / 10.0) ** 2

            gap = self._min_distance(ego.veh_id)
            min_gaps.append(gap)
            if gap < self.cfg.proximity_threshold:
                rewards[i] -= self.cfg.proximity_weight * (self.cfg.proximity_threshold - gap)
            # Forcing others into emergency braking is a real cost of an
            # uncommunicative manoeuvre, not just an aesthetic one.
            rewards[i] -= self.cfg.courtesy_weight * getattr(self, "_induced_brakes", 0)

        return rewards, dones, {"min_gap": float(np.min(min_gaps)) if min_gaps else np.inf}

    def _min_distance(self, veh_id):
        try:
            ex, ey = self.conn.vehicle.getPosition(veh_id)
        except traci.TraCIException:
            return np.inf
        best = np.inf
        for other in self.conn.vehicle.getIDList():
            if other == veh_id:
                continue
            ox, oy = self.conn.vehicle.getPosition(other)
            dist = float(np.hypot(ox - ex, oy - ey))
            best = min(best, dist)
        return best

    # ----------------------------------------------------------- observations

    def _vehicle_snapshot(self):
        snapshot = {}
        for veh_id in self.conn.vehicle.getIDList():
            x, y = self.conn.vehicle.getPosition(veh_id)
            angle = np.deg2rad(90.0 - self.conn.vehicle.getAngle(veh_id))
            speed = self.conn.vehicle.getSpeed(veh_id)
            snapshot[veh_id] = {
                "x": x,
                "y": y,
                "heading": angle,
                "speed": speed,
                "vx": speed * np.cos(angle),
                "vy": speed * np.sin(angle),
                "edge": self.conn.vehicle.getRoadID(veh_id),
                "is_ego": veh_id.startswith("ego_"),
            }
        return snapshot

    def _signed_conflict_distance(self, state):
        """Distance to the conflict point, negative once the vehicle has passed it."""
        cx, cy = self.spec.conflict_point
        approach = np.array([cx - state["x"], cy - state["y"]])
        heading_vec = np.array([np.cos(state["heading"]), np.sin(state["heading"])])
        distance = float(np.linalg.norm(approach))
        sign = 1.0 if float(np.dot(approach, heading_vec)) >= 0.0 else -1.0
        return sign * distance

    def _build_observations(self):
        snapshot = self._vehicle_snapshot()
        self._last_snapshot = snapshot
        obs = np.zeros((len(self.egos), self.obs_dim), dtype=np.float32)
        cx, cy = self.spec.conflict_point

        for i, ego in enumerate(self.egos):
            state = snapshot.get(ego.veh_id)
            if state is None:
                if ego.history:
                    obs[i] = ego.history[-1]
                continue

            signed = self._signed_conflict_distance(state)

            ego_feat = np.array([
                state["speed"] / self.spec.max_speed,
                ego.last_accel / self.cfg.max_accel,
                signed / 100.0,
                np.cos(state["heading"]),
                np.sin(state["heading"]),
                self.step_count / self.cfg.horizon,
            ], dtype=np.float32)

            neigh_feat = self._neighbor_features(state, snapshot, ego.veh_id)
            vec = np.concatenate([ego_feat, neigh_feat]).astype(np.float32)
            obs[i] = vec
            ego.history.append(vec)
            if len(ego.history) > self.cfg.history_len:
                ego.history.pop(0)
        return obs

    def _neighbor_ids(self, ego_state, snapshot, ego_id):
        items = []
        edges = {}
        for veh_id, other in snapshot.items():
            if veh_id == ego_id:
                continue
            dx, dy = other["x"] - ego_state["x"], other["y"] - ego_state["y"]
            dist = float(np.hypot(dx, dy))
            if dist > self.cfg.neighbor_radius:
                continue
            items.append((veh_id, other["x"], other["y"], other["heading"],
                          other["speed"], dist))
            edges[veh_id] = other.get("edge")
        return rank_neighbor_ids(
            items, self.cfg.n_neighbors,
            conflict_point=self.spec.conflict_point,
            prefixes=self.spec.approach_edge_prefixes, edges=edges,
            decision_distance=self.cfg.decision_distance)

    def _neighbor_features(self, ego_state, snapshot, ego_id):
        cos_h, sin_h = np.cos(-ego_state["heading"]), np.sin(-ego_state["heading"])
        feats = np.zeros(self.cfg.n_neighbors * self.neighbor_feature_dim, dtype=np.float32)
        for slot, veh_id in enumerate(self._neighbor_ids(ego_state, snapshot, ego_id)):
            other = snapshot[veh_id]
            dx, dy = other["x"] - ego_state["x"], other["y"] - ego_state["y"]
            rx = dx * cos_h - dy * sin_h
            ry = dx * sin_h + dy * cos_h
            dvx, dvy = other["vx"] - ego_state["vx"], other["vy"] - ego_state["vy"]
            rvx = dvx * cos_h - dvy * sin_h
            rvy = dvx * sin_h + dvy * cos_h
            other_to_conflict = self._signed_conflict_distance(other)
            start = slot * self.neighbor_feature_dim
            feats[start:start + self.neighbor_feature_dim] = [
                rx / 50.0,
                ry / 50.0,
                rvx / self.spec.max_speed,
                rvy / self.spec.max_speed,
                other["speed"] / self.spec.max_speed,
                other_to_conflict / 100.0,
                1.0,
            ]
        return feats

    # -------------------------------------------------------------- recording

    def _record_frame(self):
        snapshot = getattr(self, "_last_snapshot", None)
        if snapshot is None:
            return
        self._frames.append({
            "t": self.step_count * self.dt,
            "vehicles": {
                veh_id: (state["x"], state["y"], state["vx"], state["vy"], state["speed"], state["heading"], state["is_ego"])
                for veh_id, state in snapshot.items()
            },
            "edges": {veh_id: state.get("edge", "") for veh_id, state in snapshot.items()},
            "ego_accel": {ego.veh_id: ego.last_accel for ego in self.egos},
        })

    @property
    def frames(self):
        return self._frames
