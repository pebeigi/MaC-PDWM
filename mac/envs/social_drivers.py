"""Latent-type background drivers.

Each background vehicle is assigned a hidden behavioural type at insertion:

``yielder``    always concedes the conflict point,
``contester``  never concedes and will not brake for a crossing vehicle,
``reactive``   decides on approach, based on how the ego is moving.

The type is never exposed to the policy, so it has to be inferred from motion.
The reactive type closes the negotiation loop: the ego's own trajectory changes
the probability that the other driver concedes, which is what makes the future
distribution multimodal rather than merely noisy.
"""
import numpy as np

from mac.data.scene import edge_allowed

# SUMO speed-mode bits: 0 safe speed, 1/2 accel and decel bounds, 3 right of way
# for approaching foes, 4 red lights, 5 disregard right of way for foes already
# inside the junction. A contester keeps car-following safety (bit 0) but gives
# way to nobody at the junction, which is what makes it able to hit the ego.
YIELD_MODE = 31
CONTEST_MODE = 1 + 2 + 4 + 32

# The junction-model parameters that actually stop a driver from braking for a
# crossing foe live on the vehicle type, so conceding is a type switch.
YIELD_VTYPE = "background"
CONTEST_VTYPE = "bg_contest"

TYPES = ("yielder", "contester", "reactive")

# Influence-kernel coefficients (priority margin, ego intent, bias). ``beta_intent``
# is the strength of the communication channel: setting it to zero makes the
# latent type independent of the ego's motion and reduces the environment to an
# ordinary hidden-parameter POMDP, which is the no-channel control condition.
# The margin is computed from quantities already visible in the history, so a
# margin-dominated kernel is predictable without the plan; the channel only
# carries information when the intent term dominates.
BETA_MARGIN = 0.3
BETA_INTENT = 2.5
BETA_BIAS = -0.3

# Per-driver logit offset drawn at insertion and never observable. Without it the
# commitment is a deterministic function of quantities the history already
# contains, the response distribution is unimodal, and a constant-velocity
# predictor is optimal.
STUBBORNNESS_SCALE = 1.2

# A commitment must move the vehicle inside the prediction horizon, otherwise the
# interventional response is kinematically empty: switching the junction model
# alone leaves the branch separation near zero for several seconds.
CONCESSION_SPEED_FRACTION = 0.3
CONCESSION_DURATION = 2.5
CONTEST_SPEED_GAIN = 1.2
MIN_CONCESSION_SPEED = 0.5

# A background vehicle braking behind a close leader is queueing, not reacting
# to the ego, so those events must not be attributed to the ego.
QUEUE_LOOKAHEAD = 30.0
QUEUE_GAP = 25.0


class SocialDriverManager:
    def __init__(self, conflict_point, rng, type_probs=(0.1, 0.1, 0.8),
                 decision_distance=35.0, hard_brake_threshold=-3.0,
                 beta_margin=BETA_MARGIN, beta_intent=BETA_INTENT, beta_bias=BETA_BIAS,
                 intent_window=0.8, courtesy_grace_steps=3,
                 brake_attribution_distance=60.0,
                 stubbornness_scale=STUBBORNNESS_SCALE,
                 kinematic_concession=True,
                 concession_speed_fraction=CONCESSION_SPEED_FRACTION,
                 concession_duration=CONCESSION_DURATION,
                 contest_speed_gain=CONTEST_SPEED_GAIN,
                 approach_edge_prefixes=(),
                 scenario=""):
        self.conflict_point = np.asarray(conflict_point, dtype=float)
        self.rng = rng
        self.type_probs = np.asarray(type_probs, dtype=float)
        self.type_probs /= self.type_probs.sum()
        self.decision_distance = decision_distance
        self.hard_brake_threshold = hard_brake_threshold
        # How close to the conflict a braking vehicle must be for the ego to be a
        # plausible cause of that brake.
        self.brake_attribution_distance = float(brake_attribution_distance)
        self.beta_margin = beta_margin
        self.beta_intent = beta_intent
        self.beta_bias = beta_bias
        # A driver that concedes decelerates at the junction. Charging that to the
        # ego taxes successful negotiation, so those steps can be granted a grace
        # window and only later hard braking is attributed.
        self.courtesy_grace_steps = int(courtesy_grace_steps)
        self.stubbornness_scale = float(stubbornness_scale)
        self.kinematic_concession = bool(kinematic_concession)
        self.concession_speed_fraction = float(concession_speed_fraction)
        self.concession_duration = float(concession_duration)
        self.contest_speed_gain = float(contest_speed_gain)
        # Empty = any edge. Roundabout sets this so Euclidean-near vehicles on
        # the wrong ring arc (e.g. circ_NW) cannot commit as conflict partners.
        self.approach_edge_prefixes = tuple(approach_edge_prefixes)
        self.scenario = str(scenario or "")

        self.types = {}
        self.stubbornness = {}
        self._speed_influence = {}
        self.resolved = {}
        self.resolved_step = {}
        self._pending_mode = {}
        self._pending_vtype = {}
        self._prev_speed = {}
        self._braking_active = {}
        self.hard_brake_events = 0
        self.ego_induced_brakes = 0
        self.total_induced_decel = 0.0
        self.step_index = 0
        # Reactive types should respond to the *plan*, not a single control tick.
        # The EMA window must stay well below the history span H*dt, otherwise the
        # intent signal at decision time is a statistic of the history the model
        # already sees and the candidate plan carries no extra information.
        self.intent_window = float(intent_window)
        self._ego_accel_ema = 0.0
        self._ego_engaged_step = None
        self._influence_opportunities = set()

    def reset(self):
        self.types.clear()
        self.stubbornness.clear()
        self._speed_influence.clear()
        self.resolved.clear()
        self.resolved_step.clear()
        self._pending_mode.clear()
        self._pending_vtype.clear()
        self._prev_speed.clear()
        self._braking_active.clear()
        self.hard_brake_events = 0
        self.ego_induced_brakes = 0
        self.total_induced_decel = 0.0
        self.step_index = 0
        self._ego_accel_ema = 0.0
        self._ego_engaged_step = None
        self._influence_opportunities.clear()

    def register(self, veh_id):
        driver_type = TYPES[int(self.rng.choice(len(TYPES), p=self.type_probs))]
        self.types[veh_id] = driver_type
        self.stubbornness[veh_id] = float(
            self.rng.normal(0.0, self.stubbornness_scale))
        if driver_type == "yielder":
            self._pending_mode[veh_id] = YIELD_MODE
            self._pending_vtype[veh_id] = YIELD_VTYPE
        else:
            # Reactive drivers start out contesting and may concede later.
            self._pending_mode[veh_id] = CONTEST_MODE
            self._pending_vtype[veh_id] = CONTEST_VTYPE
        return driver_type

    def apply_pending(self, conn):
        if not self._pending_mode:
            return
        present = set(conn.vehicle.getIDList())
        for veh_id in list(self._pending_mode):
            if veh_id not in present:
                continue
            mode = self._pending_mode.pop(veh_id)
            vtype = self._pending_vtype.pop(veh_id, None)
            try:
                if vtype is not None:
                    conn.vehicle.setType(veh_id, vtype)
                conn.vehicle.setSpeedMode(veh_id, mode)
            except Exception:
                pass

    def _commit(self, conn, veh_id, state, decision, dt):
        """Execute a commitment as a speed change, not only a junction-model switch.

        The type switch alone is invisible for several seconds: it changes what
        the driver does *at* the junction, so two branches of the same scene stay
        within centimetres of each other across the whole prediction horizon and
        the interventional response is unidentifiable.
        """
        vtype = YIELD_VTYPE if decision == "yield" else CONTEST_VTYPE
        mode = YIELD_MODE if decision == "yield" else CONTEST_MODE
        try:
            conn.vehicle.setType(veh_id, vtype)
            conn.vehicle.setSpeedMode(veh_id, mode)
        except Exception:
            pass
        if not self.kinematic_concession:
            return
        speed = float(state.get("speed", 0.0))
        if decision == "yield":
            target = max(MIN_CONCESSION_SPEED, speed * self.concession_speed_fraction)
        else:
            target = speed * self.contest_speed_gain
            try:
                allowed = float(conn.vehicle.getAllowedSpeed(veh_id))
            except Exception:
                allowed = target
            # On the ring the vehicle is already at allowed speed, so a 1.2x
            # contest is a no-op. Hold speed instead of pretending to surge.
            if self.scenario == "roundabout":
                target = min(speed, allowed)
            else:
                target = min(target, allowed)
        self._set_speed_influence(conn, veh_id, target, dt)

    def _set_speed_influence(self, conn, veh_id, target, dt):
        try:
            conn.vehicle.slowDown(veh_id, float(target), self.concession_duration)
        except Exception:
            return
        # Kept on the manager so matched-branch state save/restore carries it:
        # a restored simulator state does not replay an in-flight speed command.
        span = int(np.ceil(self.concession_duration / max(dt, 1e-6)))
        self._speed_influence[veh_id] = {
            "speed": float(target),
            "end_step": self.step_index + span,
        }

    def _expire_influences(self, dt):
        for veh_id, entry in list(self._speed_influence.items()):
            if self.step_index >= entry["end_step"]:
                self._speed_influence.pop(veh_id, None)

    def reapply_influences(self, conn, dt):
        """Re-issue active speed commands after a simulator state restore."""
        for veh_id, entry in list(self._speed_influence.items()):
            remaining = (entry["end_step"] - self.step_index) * dt
            if remaining <= 0:
                self._speed_influence.pop(veh_id, None)
                continue
            try:
                conn.vehicle.slowDown(veh_id, entry["speed"], remaining)
            except Exception:
                self._speed_influence.pop(veh_id, None)

    def _signed_distance(self, x, y, heading):
        approach = self.conflict_point - np.array([x, y])
        heading_vec = np.array([np.cos(heading), np.sin(heading)])
        distance = float(np.linalg.norm(approach))
        return distance if float(np.dot(approach, heading_vec)) >= 0 else -distance

    def _edge_allowed(self, state):
        return edge_allowed(state.get("edge"), self.approach_edge_prefixes)

    def _in_decision_zone(self, state):
        """True when a background driver may commit.

        Requires (1) Euclidean proximity, (2) optional approach-edge allowlist,
        and (3) signed approach > 0 so post-conflict / outbound vehicles do not
        resolve. The allowlist is what keeps roundabout commits on the south
        feeders instead of the Euclidean-near opposite arc.
        """
        radial = float(np.linalg.norm(
            self.conflict_point - np.array([state["x"], state["y"]])))
        if radial > self.decision_distance:
            return False
        if not self._edge_allowed(state):
            return False
        signed = self._signed_distance(state["x"], state["y"], state["heading"])
        return signed > 0.0

    def update(self, conn, snapshot, ego_states, dt):
        """Resolve drivers that reach the decision zone and accumulate disruption stats."""
        self.step_index += 1
        self._expire_influences(dt)
        # A reactive driver is only influenceable if it is still undecided once
        # the ego is close enough for its motion to read as a claim; drivers that
        # commit before then were never negotiable.
        if self._ego_engaged_step is None:
            for ego in ego_states:
                if 0.0 < ego["d_conflict"] <= self.decision_distance:
                    self._ego_engaged_step = self.step_index
                    break
        accels = [float(e.get("accel", 0.0)) for e in ego_states]
        if accels:
            instant = float(np.mean(accels))
            alpha = float(np.clip(dt / max(self.intent_window, dt), 0.0, 1.0))
            self._ego_accel_ema = (1.0 - alpha) * self._ego_accel_ema + alpha * instant
        for veh_id, state in snapshot.items():
            if state["is_ego"] or self.resolved.get(veh_id):
                continue
            driver_type = self.types.get(veh_id)
            if driver_type is None:
                continue
            d_other = self._signed_distance(state["x"], state["y"], state["heading"])
            if not self._in_decision_zone(state):
                continue
            if driver_type == "reactive" and self._ego_engaged_step is not None:
                self._influence_opportunities.add(veh_id)

            if driver_type == "yielder":
                self._commit(conn, veh_id, state, "yield", dt)
                self._record(veh_id, "yield")
                continue
            if driver_type == "contester":
                self._commit(conn, veh_id, state, "contest", dt)
                self._record(veh_id, "contest")
                continue

            ego = self._closest_relevant_ego(ego_states)
            if ego is None:
                continue
            decision = "yield" if self._decide_yield(ego, state, d_other, veh_id) else "contest"
            self._commit(conn, veh_id, state, decision, dt)
            self._record(veh_id, decision)

    def _record(self, veh_id, decision):
        self.resolved[veh_id] = decision
        self.resolved_step[veh_id] = self.step_index

    def _closest_relevant_ego(self, ego_states):
        candidates = [ego for ego in ego_states if ego["d_conflict"] > -5.0]
        if not candidates:
            return None
        return min(candidates, key=lambda ego: ego["d_conflict"])

    def _decide_yield(self, ego, other_state, d_other, veh_id=None):
        ego_ttc = ego["d_conflict"] / max(ego["speed"], 0.5)
        other_ttc = d_other / max(other_state["speed"], 0.5)
        # Positive when the ego is clearly going to arrive first.
        priority_margin = other_ttc - ego_ttc
        # An accelerating ego is claiming the conflict point and the other driver
        # is more likely to concede; a decelerating ego is offering to give way,
        # which invites the other driver to take it.
        intent_signal = float(self._ego_accel_ema)
        logit = (self.beta_margin * priority_margin
                 + self.beta_intent * intent_signal
                 + self.beta_bias
                 + self.stubbornness.get(veh_id, 0.0))
        prob = 1.0 / (1.0 + np.exp(-logit))
        return bool(self.rng.random() < prob)

    def yield_probability(self, priority_margin, ego_accel):
        """Ground-truth channel, exposed for evaluation only (never to the agent).

        Marginal over the unobservable per-driver offset, via the standard
        logistic-normal approximation.
        """
        logit = (self.beta_margin * priority_margin
                 + self.beta_intent * ego_accel
                 + self.beta_bias)
        scale = np.sqrt(1.0 + np.pi * self.stubbornness_scale ** 2 / 8.0)
        return float(1.0 / (1.0 + np.exp(-logit / scale)))

    def _queueing(self, conn, veh_id):
        """True when the vehicle is following a close leader that is not an ego."""
        try:
            leader = conn.vehicle.getLeader(veh_id, QUEUE_LOOKAHEAD)
        except Exception:
            return False
        if not leader:
            return False
        leader_id, gap = leader
        if not leader_id or leader_id.startswith("ego_"):
            return False
        return gap < QUEUE_GAP

    def _conceding(self, veh_id):
        """True while a vehicle is executing the concession it just committed to."""
        if self.courtesy_grace_steps <= 0:
            return False
        if self.resolved.get(veh_id) != "yield":
            return False
        step = self.resolved_step.get(veh_id)
        return step is not None and self.step_index - step <= self.courtesy_grace_steps

    def _track_braking(self, conn, snapshot, ego_states, dt):
        ego_near = any(abs(ego["d_conflict"]) < 60.0 for ego in ego_states)
        for veh_id, state in snapshot.items():
            if state["is_ego"]:
                continue
            prev = self._prev_speed.get(veh_id)
            self._prev_speed[veh_id] = state["speed"]
            if prev is None:
                continue
            accel = (state["speed"] - prev) / dt
            if accel >= self.hard_brake_threshold:
                self._braking_active[veh_id] = False
                continue
            # Count one braking event at its onset, not once per decision step.
            is_new_event = not self._braking_active.get(veh_id, False)
            self._braking_active[veh_id] = True
            if not is_new_event:
                continue
            self.hard_brake_events += 1
            if self._conceding(veh_id):
                continue
            # Attribute the event to the ego only when the ego is plausibly the
            # cause: both are near the conflict point and the braking vehicle is
            # not simply following a queue. Without the braker's own distance
            # check, any hard brake anywhere on the network counts for as long as
            # the ego is on approach.
            d_other = self._signed_distance(state["x"], state["y"], state["heading"])
            braker_near = -5.0 < d_other < self.brake_attribution_distance
            if ego_near and braker_near and not self._queueing(conn, veh_id):
                self.ego_induced_brakes += 1
                self.total_induced_decel += -accel * dt

    def track_braking(self, conn, snapshot, ego_states, dt):
        """Track disruption at simulator-step resolution."""
        self._track_braking(conn, snapshot, ego_states, dt)

    def stats(self):
        counts = {name: 0 for name in TYPES}
        for driver_type in self.types.values():
            counts[driver_type] += 1
        # Only reactive drivers can be influenced, so episodes in which one of
        # them actually committed are the ones where a negotiation gain is even
        # possible. Aggregate return averages that subset away.
        engaged = self._ego_engaged_step
        def influenceable(veh_id):
            if self.types.get(veh_id) != "reactive":
                return False
            # Committed at or after the ego entered the decision zone, so the
            # ego's motion was part of the state the driver decided on.
            return engaged is not None and self.resolved_step.get(veh_id, -1) >= engaged
        reactive_resolved = sum(1 for veh_id in self.resolved if influenceable(veh_id))
        reactive_yields = sum(1 for veh_id, decision in self.resolved.items()
                              if influenceable(veh_id) and decision == "yield")
        reactive_registered = sum(
            1 for driver_type in self.types.values() if driver_type == "reactive")
        reactive_total_resolved = sum(
            1 for veh_id in self.resolved if self.types.get(veh_id) == "reactive")
        return {
            "driver_types": counts,
            "resolved": dict(self.resolved),
            "resolved_step": dict(self.resolved_step),
            "hard_brake_events": self.hard_brake_events,
            "ego_induced_brakes": self.ego_induced_brakes,
            "induced_decel": float(self.total_induced_decel),
            "reactive_resolved": reactive_resolved,
            "reactive_yields": reactive_yields,
            "reactive_registered": reactive_registered,
            "reactive_total_resolved": reactive_total_resolved,
            "reactive_unresolved": reactive_registered - reactive_total_resolved,
            "reactive_opportunities": len(self._influence_opportunities),
        }
