"""Turn raw episodes into tensors for the conditional world model.

One sample is taken at every decision step at which the ego is active:

    history   (H, 1 + K, 5)  ego and neighbour states over the past H steps
    ego_plan  (F, 3)         the ego's own future motion (the "message")
    future    (F, K, 3)      the neighbours' response to that motion
    types     (K,)           latent driver type, kept for evaluation only

All coordinates are expressed in the ego frame at the prediction time, so the
model learns relative interaction geometry rather than absolute map position.
"""
import argparse
import glob
import os
import pickle

import numpy as np

from mac.data.scene import extract_future, extract_scene
from mac.envs.sumo_planning_env import SCENARIOS

# The supervised target is what the driver actually did at the conflict, not
# which type it was drawn from: a driver that never reaches the decision point
# is genuinely ambiguous and is masked out of the loss. ``resolved`` is written
# by the environment only once a vehicle enters the decision zone, so it already
# encodes exactly that distinction for every type.
YIELD, CONTEST, UNKNOWN = 0, 1, -1


def decision_lag(veh_id, resolved_step, t, future_len):
    """Steps from the prediction time to the neighbour's commitment, or -1."""
    if resolved_step is None or t is None or future_len is None:
        return -1
    step = resolved_step.get(veh_id)
    if step is None or not (t < step <= t + future_len):
        return -1
    return int(step - t)


def intention_label(veh_id, driver_types, resolved, resolved_step=None, t=None, future_len=None):
    """Label a neighbour only when it commits inside the plan's horizon.

    ``q_phi(theta | h, u)`` is meant to answer "will this driver concede if I
    execute ``u``", so the supervision must come from decisions that actually
    fall inside the window ``u`` covers. A decision already taken at time ``t``
    is not a counterfactual, and one taken long after ``t + F`` is not caused by
    ``u``; including either teaches the head the marginal and makes it blind to
    the plan, which is exactly the failure the probe response exposes.
    """
    decision = resolved.get(veh_id)
    if decision not in ("yield", "contest"):
        return UNKNOWN
    if resolved_step is not None and t is not None and future_len is not None:
        step = resolved_step.get(veh_id)
        if step is None or not (t < step <= t + future_len):
            return UNKNOWN
    return YIELD if decision == "yield" else CONTEST


# Behaviour policies whose acceleration sequence is chosen before the episode
# and never reacts to the scene. For these episodes u is independent of h, so a
# conditional fitted on them is p(y | h, do(u)) rather than the observational
# p(y | h, u); the gap-acceptance episodes are reactive and confound the two.
INTERVENTIONAL_POLICIES = ("constant", "hold", "plan")
CONFLICT_POINTS = {
    name: tuple(spec.conflict_point) for name, spec in SCENARIOS.items()
}


def is_interventional(record):
    if "interventional" in record:
        return bool(record["interventional"])
    tag = str(record.get("policy", ""))
    return any(tag.startswith(p) for p in INTERVENTIONAL_POLICIES)


def priority_margins(record, t, ego_id, neighbor_ids, n_neighbors):
    """Scene-conditional TTC margin used by the simulator's intent kernel."""
    out = np.zeros(n_neighbors, dtype=np.float32)
    point = np.asarray(CONFLICT_POINTS.get(record.get("scenario", "cross"),
                                           (0.0, 0.0)))
    vehicles = record["frames"][t]["vehicles"]

    def signed_ttc(state):
        x, y, _, _, speed, heading = state[:6]
        approach = point - np.asarray([x, y])
        sign = 1.0 if float(np.dot(
            approach, [np.cos(heading), np.sin(heading)])) >= 0 else -1.0
        return sign * float(np.linalg.norm(approach)) / max(float(speed), 0.5)

    ego_state = vehicles.get(ego_id)
    if ego_state is None:
        return out
    ego_ttc = signed_ttc(ego_state)
    for k, veh_id in enumerate(neighbor_ids):
        state = vehicles.get(veh_id)
        if state is not None:
            out[k] = signed_ttc(state) - ego_ttc
    return out


def build_episode(record, history_len, future_len, n_neighbors):
    frames = record["frames"]
    ego_ids = record["ego_ids"]
    if not ego_ids:
        return []
    ego_id = ego_ids[0]
    interventional = is_interventional(record)
    driver_types = record["driver_types"]
    resolved = record.get("resolved", {})
    resolved_step = record.get("resolved_step", {})

    samples = []
    n_frames = len(frames)
    selected_times = record.get("sample_times")
    selected_times = set(selected_times) if selected_times is not None else None
    conflict_point = CONFLICT_POINTS.get(record.get("scenario", "cross"), (0.0, 0.0))
    spec = SCENARIOS.get(record.get("scenario", "cross"))
    prefixes = tuple(spec.approach_edge_prefixes) if spec is not None else ()
    decision_distance = None if spec is None else spec.decision_distance
    for t in range(history_len - 1, n_frames - future_len):
        if selected_times is not None and t not in selected_times:
            continue
        scene = extract_scene(
            frames, t, ego_id, history_len, n_neighbors,
            conflict_point=conflict_point,
            approach_edge_prefixes=prefixes,
            decision_distance=decision_distance)
        if scene is None or not scene["neighbor_ids"]:
            continue

        ego_plan, future = extract_future(
            frames, t, ego_id, scene["neighbor_ids"], future_len,
            scene["origin"], scene["rot"], n_neighbors,
        )

        # Predict displacement from where each neighbour is now; absolute
        # ego-frame positions would force the model to spend capacity on the
        # scene layout it already receives as input.
        current = scene["history"][-1, 1:, :2]
        future[..., :2] -= current[None, :, :] * (future[..., 2:] > 0)

        types = np.full(n_neighbors, UNKNOWN, dtype=np.int64)
        lags = np.full(n_neighbors, -1, dtype=np.int64)
        for k, veh_id in enumerate(scene["neighbor_ids"]):
            types[k] = intention_label(veh_id, driver_types, resolved,
                                       resolved_step, t, future_len)
            lags[k] = decision_lag(veh_id, resolved_step, t, future_len)

        samples.append({
            "history": scene["history"],
            "ego_plan": ego_plan,
            "future": future,
            "types": types,
            "lags": lags,
            "interventional": interventional,
            "intervention_id": record.get("intervention_id"),
            "probe_accel": float(record.get("probe_accel", np.nan)),
            "priority_margin": priority_margins(
                record, t, ego_id, scene["neighbor_ids"], n_neighbors),
        })
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/mac/raw/cross")
    parser.add_argument("--out", default="data/mac/cross.npz")
    parser.add_argument("--history_len", type=int, default=5)
    # A driver's commitment is decided over several seconds of consistent ego
    # motion. At 10 steps (4 s) an assertive and a yielding plan produce the
    # same commitment for two thirds of neighbours; at 20 steps (8 s) they
    # differ for most of them, which is the signal the world model exists to
    # capture.
    parser.add_argument("--future_len", type=int, default=20)
    parser.add_argument("--n_neighbors", type=int, default=5)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.pkl")))
    if not paths:
        raise SystemExit(f"no episodes found in {args.raw_dir}")

    # Consecutive decision steps inside one episode produce almost identical
    # overlapping windows, so a random split over samples would put near-copies
    # of the same interaction on both sides. The split is therefore over
    # episodes, decided before any sample is generated.
    records = []
    split_groups = []
    for i, path in enumerate(paths):
        with open(path, "rb") as handle:
            record = pickle.load(handle)
        records.append(record)
        intervention_id = record.get("intervention_id")
        split_groups.append(
            f"pair:{intervention_id}" if intervention_id is not None
            else f"episode:{i}")
    unique_groups = sorted(set(split_groups))
    rng = np.random.default_rng(0)
    group_perm = rng.permutation(len(unique_groups))
    n_val_episodes = max(1, int(len(unique_groups) * args.val_fraction))
    val_groups = {unique_groups[i] for i in group_perm[:n_val_episodes]}

    histories, plans, futures, types, lags, episode_idx = [], [], [], [], [], []
    interventional = []
    intervention_keys, probe_accels, priority_margin, sample_groups = [], [], [], []
    for i, (path, record) in enumerate(zip(paths, records)):
        for sample in build_episode(record, args.history_len, args.future_len, args.n_neighbors):
            histories.append(sample["history"])
            plans.append(sample["ego_plan"])
            futures.append(sample["future"])
            types.append(sample["types"])
            lags.append(sample["lags"])
            interventional.append(sample["interventional"])
            episode_idx.append(i)
            intervention_keys.append(sample["intervention_id"])
            probe_accels.append(sample["probe_accel"])
            priority_margin.append(sample["priority_margin"])
            sample_groups.append(split_groups[i])
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(paths)} episodes, {len(histories)} samples", flush=True)

    history = np.stack(histories)
    ego_plan = np.stack(plans)
    future = np.stack(futures)
    type_arr = np.stack(types)
    lag_arr = np.stack(lags)
    episode_arr = np.asarray(episode_idx)
    interventional_arr = np.asarray(interventional, dtype=bool)
    pair_keys = sorted({key for key in intervention_keys if key is not None})
    pair_lookup = {key: i for i, key in enumerate(pair_keys)}
    pair_arr = np.asarray(
        [pair_lookup.get(key, -1) for key in intervention_keys], dtype=np.int64)
    probe_arr = np.asarray(probe_accels, dtype=np.float32)
    margin_arr = np.stack(priority_margin)

    is_val = np.asarray([group in val_groups for group in sample_groups])
    val_idx = np.flatnonzero(is_val)
    train_idx = np.flatnonzero(~is_val)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        history_train=history[train_idx], ego_plan_train=ego_plan[train_idx],
        future_train=future[train_idx], types_train=type_arr[train_idx],
        history_val=history[val_idx], ego_plan_val=ego_plan[val_idx],
        future_val=future[val_idx], types_val=type_arr[val_idx],
        episode_train=episode_arr[train_idx], episode_val=episode_arr[val_idx],
        lags_train=lag_arr[train_idx], lags_val=lag_arr[val_idx],
        interventional_train=interventional_arr[train_idx],
        interventional_val=interventional_arr[val_idx],
        pair_id_train=pair_arr[train_idx], pair_id_val=pair_arr[val_idx],
        probe_accel_train=probe_arr[train_idx], probe_accel_val=probe_arr[val_idx],
        priority_margin_train=margin_arr[train_idx],
        priority_margin_val=margin_arr[val_idx],
    )
    labelled = int((type_arr >= 0).sum())
    do_labelled = int((type_arr[interventional_arr] >= 0).sum())
    print(f"wrote {args.out}: {len(train_idx)} train / {len(val_idx)} val samples "
          f"from {len(paths)} files / {len(unique_groups)} split groups "
          f"({len(unique_groups) - n_val_episodes} train / "
          f"{n_val_episodes} val groups); "
          f"{labelled}/{type_arr.size} neighbour slots labelled, "
          f"{do_labelled} of them on open-loop (interventional) episodes")


if __name__ == "__main__":
    main()
