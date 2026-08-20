"""How much of a counterfactual belief feature is signal, and how much is sampler noise?

The influence features (shift, risk contrast) are differences between two probes.
Each probe is a Monte-Carlo mean over ``n_samples`` diffusion draws. If the two
probes are drawn with independent noise, the difference carries the plan effect
*plus* two independent Monte-Carlo errors. This script estimates both terms and
reports the correlation of the cheap estimator with a high-sample reference.

``--common-noise`` couples the probes (same latent draw, different action), which
is the textbook definition of a counterfactual contrast and cancels the shared
noise term.
"""
import argparse

import numpy as np
import torch

from mac.data.normalize import decode_samples
from mac.data.scene import synthetic_plan
from mac.models.diffusion_world_model import DiffusionWorldModel


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = DiffusionWorldModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["pos_scale"], ckpt["vel_scale"]


def probe_batch(history_raw, speeds, accels, future_len, dt, max_speed,
                pos_scale, vel_scale, device):
    """(n_scenes * n_probes) history/plan tensors, probe index varying fastest."""
    n, p = history_raw.shape[0], len(accels)
    hist = history_raw.repeat_interleave(p, dim=0).clone()
    hist[..., :2] /= pos_scale
    hist[..., 2:4] /= vel_scale
    plans = np.stack([
        synthetic_plan(float(speeds[i]), float(a), future_len, dt, max_speed)
        for i in range(n) for a in accels
    ])
    plans[..., :2] /= pos_scale
    plans[..., 2] /= vel_scale
    return hist, torch.from_numpy(plans).float().to(device)


def shift_and_risk(samples, valid, conflict_radius, ego_xy):
    """Per-probe influence shift (vs hold) and collision risk, as in BeliefEncoder."""
    pred_mean = samples.mean(dim=1)                     # (N*P, F, K, 2)
    dist = torch.linalg.norm(samples - ego_xy[:, None, :, None, :], dim=-1)
    dist = dist + (1.0 - valid)[:, None, None, :] * 1e3
    per_sample_min = dist.min(dim=-1).values.min(dim=-1).values   # (N*P, S)
    risk = (per_sample_min < conflict_radius).float().mean(dim=-1)
    return pred_mean, risk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="data/mac/world_model_cross.pt")
    ap.add_argument("--dataset", default="data/mac/cross.npz")
    ap.add_argument("--scenes", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--reference-samples", type=int, default=512)
    ap.add_argument("--sample-steps", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.4)
    ap.add_argument("--max-speed", type=float, default=13.89)
    ap.add_argument("--conflict-radius", type=float, default=10.0)
    ap.add_argument("--accels", default="-4,0,2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    accels = [float(x) for x in args.accels.split(",")]
    hold_i, assert_i = accels.index(0.0), len(accels) - 1
    device = torch.device(args.device)
    model, pos_scale, vel_scale = load_model(args.model, device)

    data = np.load(args.dataset)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(data["history_val"]), size=args.scenes, replace=False)
    hist_np = data["history_val"][idx]
    history_raw = torch.from_numpy(hist_np).float().to(device)
    speeds = np.linalg.norm(hist_np[:, -1, 0, 2:4], axis=-1)

    hist, plans = probe_batch(history_raw, speeds, accels, model.future_len,
                              args.dt, args.max_speed, pos_scale, vel_scale, device)
    hist_dec = history_raw.repeat_interleave(len(accels), dim=0)
    valid = (hist_dec[:, -1, 1:, 4] > 0).float()
    ego_xy = torch.from_numpy(np.stack([
        synthetic_plan(float(speeds[i]), float(a), model.future_len,
                       args.dt, args.max_speed)[:, :2]
        for i in range(len(speeds)) for a in accels
    ])).float().to(device)

    def run(n_samples, common_noise, seed):
        torch.manual_seed(seed)
        s = model.sample(hist, plans, n_samples=n_samples, steps=args.sample_steps,
                         common_noise=common_noise)
        s = decode_samples(s, hist_dec)
        return shift_and_risk(s, valid, args.conflict_radius, ego_xy)

    def contrast(pred_mean, risk):
        n, p = len(speeds), len(accels)
        pm = pred_mean.view(n, p, *pred_mean.shape[1:])
        rk = risk.view(n, p)
        d = torch.linalg.norm(pm[:, assert_i] - pm[:, hold_i], dim=-1)  # (N,F,K)
        v = valid.view(n, p, -1)[:, hold_i]
        shift = (d.mean(dim=1) * v).sum(dim=1) / v.sum(dim=1).clamp(min=1)
        return shift.cpu().numpy(), (rk[:, assert_i] - rk[:, hold_i]).cpu().numpy()

    ref_shift, ref_risk = contrast(*run(args.reference_samples, True, 0))
    print(f"reference ({args.reference_samples} coupled draws): "
          f"shift mean={ref_shift.mean():.3f} m  sd={ref_shift.std():.3f}  "
          f"risk-contrast sd={ref_risk.std():.3f}")

    for label, common in (("independent (current)", False), ("common noise", True)):
        shifts, risks, corrs = [], [], []
        for rep in range(5):
            s, r = contrast(*run(args.n_samples, common, 100 + rep))
            shifts.append(s)
            risks.append(r)
            corrs.append(np.corrcoef(s, ref_shift)[0, 1])
        shifts, risks = np.stack(shifts), np.stack(risks)
        # Spread across repeats at fixed scene = pure estimator noise.
        noise = shifts.std(axis=0).mean()
        signal = shifts.mean(axis=0).std()
        print(f"\n{label}  (n_samples={args.n_samples})")
        print(f"  shift: mean={shifts.mean():.3f} m  across-scene sd={signal:.3f}  "
              f"per-scene noise sd={noise:.3f}  SNR={signal / max(noise, 1e-9):.2f}")
        print(f"  corr with reference shift: {np.mean(corrs):.3f}")
        print(f"  risk contrast: per-scene noise sd={risks.std(axis=0).mean():.3f}  "
              f"across-scene sd={risks.mean(axis=0).std():.3f}")


if __name__ == "__main__":
    main()
