"""Target construction for the world model.

Background traffic is well described by a constant-velocity roll-out, so the
model predicts the *residual* from that roll-out. All of the behaviour worth
modelling -- braking to yield, holding speed to contest -- lives in that
residual, and the target is well scaled instead of being dominated by the
trivial component of the motion.
"""
import numpy as np
import torch

POS_SCALE = 30.0    # metres, for history positions
VEL_SCALE = 15.0    # m/s, for velocities
RESID_SCALE = 8.0   # metres, for the predicted residual
DT = 0.4            # seconds per decision step


def cv_rollout(velocity, future_len, dt=DT):
    """Constant-velocity displacement for each neighbour: (..., K, 2) -> (..., F, K, 2)."""
    if isinstance(velocity, np.ndarray):
        steps = np.arange(1, future_len + 1, dtype=np.float32) * dt
        return velocity[..., None, :, :] * steps[:, None, None]
    steps = torch.arange(1, future_len + 1, device=velocity.device, dtype=velocity.dtype) * dt
    return velocity.unsqueeze(-3) * steps[:, None, None]


def encode_target(history_raw, future_raw):
    """Raw future displacement -> normalised residual target."""
    velocity = history_raw[:, -1, 1:, 2:4]
    future_len = future_raw.shape[1]
    residual = future_raw[..., :2] - cv_rollout(velocity, future_len)
    return residual / RESID_SCALE


def decode_samples(samples, history_raw):
    """Normalised residual samples -> ego-frame positions in metres.

    ``samples`` is (B, S, F, K, 2); ``history_raw`` is (B, H, 1+K, 5).
    """
    velocity = history_raw[:, -1, 1:, 2:4]
    current = history_raw[:, -1, 1:, :2]
    future_len = samples.shape[2]
    cv = cv_rollout(velocity, future_len).unsqueeze(1)
    return current[:, None, None, :, :] + cv + samples * RESID_SCALE


def normalize_inputs(history_raw, ego_plan_raw):
    history = history_raw.clone()
    ego_plan = ego_plan_raw.clone()
    history[..., :2] /= POS_SCALE
    history[..., 2:4] /= VEL_SCALE
    ego_plan[..., :2] /= POS_SCALE
    ego_plan[..., 2] /= VEL_SCALE
    return history, ego_plan
