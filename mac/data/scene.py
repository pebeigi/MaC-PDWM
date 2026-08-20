"""Shared scene extraction.

Both the offline dataset builder and the online policy build world-model inputs
through these functions, so the geometry seen at training time and at inference
time cannot drift apart.
"""
import numpy as np

NEIGHBOR_RADIUS = 80.0
# Vehicles this far past the commit radius cannot still commit inside the
# queried horizon, and filling K=5 with them (a W_in queue on the roundabout)
# drowns the intention labels.
DECISION_RANK_MARGIN = 20.0


def ego_rotation(heading):
    cos_h, sin_h = np.cos(-heading), np.sin(-heading)
    return np.array([[cos_h, -sin_h], [sin_h, cos_h]])


def edge_allowed(edge, prefixes=()):
    """True when ``edge`` may be treated as a conflict partner.

    Empty prefixes = any edge (cross / merge). Roundabout passes the south-feeder
    allowlist so ``circ_NW`` cannot occupy a K=5 slot.
    """
    if not prefixes:
        return True
    edge = str(edge or "")
    return any(edge.startswith(prefix) for prefix in prefixes)


def _signed_to_conflict(x, y, heading, conflict_point):
    approach = np.asarray(conflict_point, dtype=float) - np.array([x, y], dtype=float)
    heading_vec = np.array([np.cos(heading), np.sin(heading)])
    radial = float(np.linalg.norm(approach))
    if radial <= 0.0:
        return 0.0
    return radial if float(np.dot(approach, heading_vec)) >= 0.0 else -radial


def rank_neighbor_ids(items, n_neighbors, conflict_point=None, prefixes=(),
                      edges=None, decision_distance=None):
    """Rank candidate neighbours; ``items`` is ``(veh_id, x, y, heading, speed, dist)``.

    When ``prefixes`` is set and ``edges`` is a dict, vehicles off the approach
    allowlist are dropped rather than merely down-ranked. When
    ``decision_distance`` is set, approaching vehicles inside
    ``D + DECISION_RANK_MARGIN`` occupy the first slots so a far queue cannot
    bury the driver who is about to commit.
    """
    apply_edge = bool(prefixes) and edges is not None
    zone = None if decision_distance is None else (
        float(decision_distance) + DECISION_RANK_MARGIN)
    ranked = []
    for veh_id, x, y, heading, speed, dist in items:
        if apply_edge and not edge_allowed(edges.get(veh_id), prefixes):
            continue
        if conflict_point is None:
            rank = (dist,)
        else:
            signed = _signed_to_conflict(x, y, heading, conflict_point)
            approaching = signed > 0.0
            ttc = signed / max(float(speed), 0.5) if approaching else 1e3
            if zone is None:
                band = 0 if approaching else 1
            else:
                # Prefer the driver about to hit the zone; keep the rest so
                # later-horizon commits are still labelled, just not in slot 0.
                in_play = approaching and signed <= zone
                band = 0 if in_play else (1 if approaching else 2)
            rank = (band, ttc, dist)
        ranked.append((rank, veh_id))
    ranked.sort(key=lambda item: item[0])
    return [veh_id for _, veh_id in ranked[:n_neighbors]]


def extract_scene(frames, t, ego_id, history_len, n_neighbors, radius=NEIGHBOR_RADIUS,
                  conflict_point=None, approach_edge_prefixes=(),
                  decision_distance=None):
    """Build the history tensor at frame ``t`` in the ego frame."""
    if t < 0 or t >= len(frames):
        return None
    vehicles = frames[t]["vehicles"]
    if ego_id not in vehicles:
        return None

    ex, ey, _, _, _, eheading, _ = vehicles[ego_id]
    rot = ego_rotation(eheading)
    origin = np.array([ex, ey])

    items = []
    for veh_id, state in vehicles.items():
        if veh_id == ego_id:
            continue
        dist = float(np.hypot(state[0] - ex, state[1] - ey))
        if dist > radius:
            continue
        items.append((veh_id, state[0], state[1], state[5], state[4], dist))
    neighbor_ids = rank_neighbor_ids(
        items, n_neighbors, conflict_point=conflict_point,
        prefixes=approach_edge_prefixes, edges=frames[t].get("edges"),
        decision_distance=decision_distance)

    history = np.zeros((history_len, 1 + n_neighbors, 5), dtype=np.float32)
    for h in range(history_len):
        idx = t - history_len + 1 + h
        if idx < 0:
            idx = 0
        past = frames[idx]["vehicles"]
        for slot, veh_id in enumerate([ego_id] + neighbor_ids):
            if veh_id not in past:
                continue
            px, py, pvx, pvy = past[veh_id][:4]
            pos = (np.array([px, py]) - origin) @ rot.T
            vel = np.array([pvx, pvy]) @ rot.T
            history[h, slot] = [pos[0], pos[1], vel[0], vel[1], 1.0]

    return {"history": history, "neighbor_ids": neighbor_ids, "origin": origin, "rot": rot}


def extract_future(frames, t, ego_id, neighbor_ids, future_len, origin, rot, n_neighbors):
    """Ego plan and neighbour response over the prediction horizon."""
    ego_plan = np.zeros((future_len, 3), dtype=np.float32)
    future = np.zeros((future_len, n_neighbors, 3), dtype=np.float32)
    for f in range(future_len):
        idx = t + 1 + f
        if idx >= len(frames):
            if f > 0:
                ego_plan[f] = ego_plan[f - 1]
            continue
        ahead = frames[idx]["vehicles"]
        if ego_id in ahead:
            px, py = ahead[ego_id][:2]
            pos = (np.array([px, py]) - origin) @ rot.T
            ego_plan[f] = [pos[0], pos[1], ahead[ego_id][4]]
        elif f > 0:
            ego_plan[f] = ego_plan[f - 1]
        for k, veh_id in enumerate(neighbor_ids):
            if veh_id not in ahead:
                continue
            px, py = ahead[veh_id][:2]
            pos = (np.array([px, py]) - origin) @ rot.T
            future[f, k] = [pos[0], pos[1], 1.0]
    return ego_plan, future


MAX_YAW_RATE = 0.6  # rad/s; ~a 20 m radius taken at 12 m/s, well above road noise
# Below this the measured turn is lane-keeping jitter or a one-off lateral nudge
# rather than a sustained arc, and extrapolating it over the horizon is worse
# than assuming a straight probe. Circulating a 20 m roundabout is ~0.4 rad/s.
MIN_YAW_RATE = 0.15


def ego_yaw_rate(history, dt):
    """Recent turn rate of the ego, in rad/s, from its own history track.

    The training plans are logged trajectories, so on curved geometry (a merge
    ramp, a roundabout) they carry real lateral motion in the ego frame. A probe
    that always drives straight would query the world model off the distribution
    it was trained on. Returns 0 for a straight approach, which is the exact
    behaviour on the crossing.
    """
    ego = np.asarray(history)[:, 0]
    valid = np.flatnonzero(ego[:, 4] > 0)
    if valid.size < 2:
        return 0.0
    first, last = ego[valid[0]], ego[valid[-1]]
    span = float(valid[-1] - valid[0]) * dt
    if span <= 0:
        return 0.0
    a0 = np.arctan2(first[3], first[2])
    a1 = np.arctan2(last[3], last[2])
    if not np.isfinite(a0) or not np.isfinite(a1):
        return 0.0
    if np.hypot(*first[2:4]) < 0.5 or np.hypot(*last[2:4]) < 0.5:
        return 0.0
    delta = float(np.arctan2(np.sin(a1 - a0), np.cos(a1 - a0)))
    omega = delta / span
    if abs(omega) < MIN_YAW_RATE:
        return 0.0
    # A noisy heading estimate extrapolated over the whole horizon would bend the
    # probe into a shape the ego cannot drive.
    return float(np.clip(omega, -MAX_YAW_RATE, MAX_YAW_RATE))


def synthetic_plan(speed, accel, future_len, dt, max_speed, yaw_rate=0.0):
    """Ego-frame trajectory implied by holding a constant acceleration.

    Used at inference time to ask the world model what the neighbours would do
    if the ego were to yield, hold, or assert. ``yaw_rate`` continues the ego's
    current turn; at 0 this is a straight roll-out along +x.
    """
    plan = np.zeros((future_len, 3), dtype=np.float32)
    x, y, v, theta = 0.0, 0.0, float(speed), 0.0
    omega = float(yaw_rate)
    for f in range(future_len):
        v = float(np.clip(v + accel * dt, 0.0, max_speed))
        theta += omega * dt
        # The ego frame has +x along the current heading.
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        plan[f] = [x, y, v]
    return plan
