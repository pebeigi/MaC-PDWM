"""Train the conditional diffusion world model on collected interactions."""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mac.data.normalize import (POS_SCALE, RESID_SCALE, VEL_SCALE, decode_samples,
                                encode_target, normalize_inputs)
from mac.models.diffusion_world_model import DiffusionWorldModel


def load_split(data, split):
    """Returns normalised inputs, the residual target, and the raw history.

    The raw history is kept so samples can be decoded back into metres.
    """
    history_raw = torch.from_numpy(data[f"history_{split}"]).float()
    ego_plan_raw = torch.from_numpy(data[f"ego_plan_{split}"]).float()
    future_raw = torch.from_numpy(data[f"future_{split}"]).float()
    types = torch.from_numpy(data[f"types_{split}"]).long()
    key = f"interventional_{split}"
    if key in data:
        interventional = torch.from_numpy(data[key]).bool()
    else:  # datasets built before the open-loop flag was recorded
        interventional = torch.ones(len(types), dtype=torch.bool)

    history, ego_plan = normalize_inputs(history_raw, ego_plan_raw)
    target = encode_target(history_raw, future_raw)
    future = torch.cat([target, future_raw[..., 2:]], dim=-1)
    return TensorDataset(history, ego_plan, future, types, interventional,
                         history_raw, future_raw)


def evaluate(model, loader, device, drop_plan=False,
             interventional_trajectory=False, type_weight=1.0,
             labelled_traj_weight=1.0):
    model.eval()
    totals, counts = 0.0, 0
    intent_correct, intent_total = 0, 0
    with torch.no_grad():
        for history, ego_plan, future, types, interventional, _, _ in loader:
            history, ego_plan = history.to(device), ego_plan.to(device)
            if drop_plan:
                ego_plan = torch.zeros_like(ego_plan)
            future, types = future.to(device), types.to(device)
            interventional = interventional.to(device)
            mask = interventional if interventional_trajectory else None
            loss, parts = model.loss(
                history, ego_plan, future, types, type_weight=type_weight,
                intent_mask=interventional, trajectory_mask=mask,
                labelled_traj_weight=labelled_traj_weight)
            totals += parts["diffusion"] * history.shape[0]
            counts += history.shape[0]

            probs = model.predict_intentions(history, ego_plan)
            valid = types >= 0
            if valid.any():
                pred = probs.argmax(dim=-1)
                intent_correct += int((pred[valid] == types[valid]).sum())
                intent_total += int(valid.sum())
    model.train()
    accuracy = intent_correct / max(intent_total, 1)
    return totals / max(counts, 1), accuracy


def paired_loader(dataset, pair_ids, probe_accels, batch_size, shuffle):
    """Build matched branch pairs, using the hold probe as reference."""
    pair_ids = np.asarray(pair_ids)
    probe_accels = np.asarray(probe_accels)
    left, right = [], []
    for pair_id in np.unique(pair_ids[pair_ids >= 0]):
        members = np.flatnonzero(pair_ids == pair_id)
        if len(members) < 2:
            continue
        ref = members[int(np.argmin(np.abs(probe_accels[members])))]
        for member in members:
            if member != ref:
                left.append(ref)
                right.append(member)
    if not left:
        return None
    tensors = dataset.tensors
    pair_set = TensorDataset(
        tensors[0][left], tensors[1][left], tensors[2][left], tensors[3][left],
        tensors[0][right], tensors[1][right], tensors[2][right], tensors[3][right])
    return DataLoader(
        pair_set, batch_size=min(batch_size, len(pair_set)),
        shuffle=shuffle)


def evaluate_pairs(model, loader, device, labelled_traj_weight=1.0):
    if loader is None or getattr(model, "independent", False):
        return 0.0
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [value.to(device) for value in batch]
            ha, pa, fa, ta, hb, pb, fb, tb = batch
            loss = model.counterfactual_delta_loss(
                ha, pa, fa, hb, pb, fb, ta, tb,
                labelled_traj_weight=labelled_traj_weight)
            total += float(loss) * batch[0].shape[0]
            count += batch[0].shape[0]
    model.train()
    return total / max(count, 1)


def displacement_error(model, loader, device, n_samples=8, max_batches=20, steps=25,
                       drop_plan=False):
    """Minimum-over-samples final displacement error, in metres."""
    model.eval()
    min_fde, mean_fde, count = 0.0, 0.0, 0
    with torch.no_grad():
        for i, (history, ego_plan, future, _, _, history_raw, future_raw) in enumerate(loader):
            if i >= max_batches:
                break
            history, ego_plan = history.to(device), ego_plan.to(device)
            if drop_plan:
                ego_plan = torch.zeros_like(ego_plan)
            history_raw, future_raw = history_raw.to(device), future_raw.to(device)
            samples = model.sample(history, ego_plan, n_samples=n_samples, steps=steps)

            predicted = decode_samples(samples, history_raw)
            current = history_raw[:, -1, 1:, :2]
            target = (future_raw[..., :2] + current[:, None, :, :]).unsqueeze(1)
            mask = future_raw[..., 2].unsqueeze(1)

            err = torch.linalg.norm(predicted - target, dim=-1) * mask
            per_sample = err[:, :, -1, :].sum(dim=-1) / mask[:, :, -1, :].sum(dim=-1).clamp(min=1)
            min_fde += float(per_sample.min(dim=1).values.sum())
            mean_fde += float(per_sample.mean(dim=1).sum())
            count += history.shape[0]
    model.train()
    return min_fde / max(count, 1), mean_fde / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/mac/cross.npz")
    parser.add_argument("--out", default="data/mac/world_model_cross.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=50)
    # History-only ablation: zero the plan input so the model learns p(y | h).
    parser.add_argument("--drop_plan", action="store_true",
                        help="train a history-only world model (plan input zeroed)")
    parser.add_argument("--independent", action="store_true",
                        help="denoise each neighbour independently (no joint "
                             "residual; implies --flat_denoiser)")
    parser.add_argument("--token_conditioned", action="store_true", default=True,
                        help="use masked per-neighbour tokens with joint attention")
    parser.add_argument("--flat_denoiser", dest="token_conditioned",
                        action="store_false",
                        help="ablation: legacy max-pooled flattened denoiser")
    parser.add_argument("--plan_dropout", type=float, default=0.1,
                        help="probability of zeroing the plan during training, so "
                             "the model also learns the history-only branch")
    parser.add_argument("--type_weight", type=float, default=1.0,
                        help="weight on the intention head relative to the "
                             "denoising loss")
    parser.add_argument("--interventional_intent", action="store_true", default=True,
                        help="train the intention head only on open-loop episodes, "
                             "where the plan is independent of the scene")
    parser.add_argument("--no_interventional_intent", dest="interventional_intent",
                        action="store_false",
                        help="ablation: fit the head on all episodes, i.e. the "
                             "observational conditional")
    parser.add_argument("--interventional_trajectory", action="store_true",
                        default=True,
                        help="fit p(y|h,do(u)) using only exogenous-plan rows")
    parser.add_argument("--all_trajectory", dest="interventional_trajectory",
                        action="store_false",
                        help="ablation: include confounded reactive-policy rows")
    parser.add_argument("--delta_weight", type=float, default=0.5,
                        help="weight for matched-branch response-delta loss")
    parser.add_argument("--labelled_traj_weight", type=float, default=1.0,
                        help="multiply the diffusion/delta loss on neighbour "
                             "slots that have an intention label")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    # Token attention is joint over neighbours; the independent ablation
    # denoises each slot with K=1, so it cannot use that denoiser.
    if args.independent:
        args.token_conditioned = False

    data = np.load(args.dataset)
    train_set = load_split(data, "train")
    val_set = load_split(data, "val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    train_pairs = paired_loader(
        train_set, data["pair_id_train"] if "pair_id_train" in data else
        np.full(len(train_set), -1),
        data["probe_accel_train"] if "probe_accel_train" in data else
        np.full(len(train_set), np.nan), args.batch_size, True)
    val_pairs = paired_loader(
        val_set, data["pair_id_val"] if "pair_id_val" in data else
        np.full(len(val_set), -1),
        data["probe_accel_val"] if "probe_accel_val" in data else
        np.full(len(val_set), np.nan), args.batch_size, False)
    if args.drop_plan or args.independent:
        # History-only has no plan contrast; independent denoises each slot
        # alone, so matched-branch joint deltas are undefined.
        train_pairs = None
        val_pairs = None

    history_len = train_set.tensors[0].shape[1]
    n_neighbors = train_set.tensors[0].shape[2] - 1
    plan_len = train_set.tensors[1].shape[1]
    future_len = train_set.tensors[2].shape[1]

    device = torch.device(args.device)
    model = DiffusionWorldModel(
        history_len=history_len, future_len=future_len, n_neighbors=n_neighbors,
        plan_len=plan_len, n_steps=args.n_steps, independent=args.independent,
        token_conditioned=args.token_conditioned,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_do = int(train_set.tensors[4].sum())
    print(f"train={len(train_set)} val={len(val_set)} device={device} "
          f"H={history_len} F={future_len} K={n_neighbors} "
          f"drop_plan={args.drop_plan} independent={args.independent} "
          f"token_conditioned={args.token_conditioned} "
          f"type_weight={args.type_weight} "
          f"interventional_intent={args.interventional_intent} "
          f"interventional_trajectory={args.interventional_trajectory} "
          f"paired_batches={0 if train_pairs is None else len(train_pairs)} "
          f"labelled_traj_weight={args.labelled_traj_weight} "
          f"({n_do}/{len(train_set)} open-loop samples)")

    best = float("inf")
    for epoch in range(args.epochs):
        started = time.time()
        running = 0.0
        pair_iter = iter(train_pairs) if train_pairs is not None else None
        for history, ego_plan, future, types, interventional, _, _ in train_loader:
            history, ego_plan = history.to(device), ego_plan.to(device)
            if args.drop_plan:
                ego_plan = torch.zeros_like(ego_plan)
            elif args.plan_dropout > 0:
                # Occasionally hiding the plan gives the model a well-defined
                # history-only branch, which is what guidance extrapolates from
                # at query time.
                keep = (torch.rand(ego_plan.shape[0], device=device)
                        >= args.plan_dropout).float()
                ego_plan = ego_plan * keep[:, None, None]
            future, types = future.to(device), types.to(device)
            mask = interventional.to(device) if args.interventional_intent else None
            loss, _ = model.loss(history, ego_plan, future, types,
                                 type_weight=args.type_weight, intent_mask=mask,
                                 trajectory_mask=(
                                     interventional.to(device)
                                     if args.interventional_trajectory else None),
                                 labelled_traj_weight=args.labelled_traj_weight)
            if pair_iter is not None and args.delta_weight > 0:
                try:
                    pair_batch = next(pair_iter)
                except StopIteration:
                    pair_iter = iter(train_pairs)
                    pair_batch = next(pair_iter)
                pair_batch = [value.to(device) for value in pair_batch]
                ha, pa, fa, ta, hb, pb, fb, tb = pair_batch
                loss = loss + args.delta_weight * model.counterfactual_delta_loss(
                    ha, pa, fa, hb, pb, fb, ta, tb,
                    labelled_traj_weight=args.labelled_traj_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * history.shape[0]
        scheduler.step()

        val_loss, intent_acc = evaluate(
            model, val_loader, device, drop_plan=args.drop_plan,
            interventional_trajectory=args.interventional_trajectory,
            type_weight=args.type_weight,
            labelled_traj_weight=args.labelled_traj_weight)
        val_delta = evaluate_pairs(
            model, val_pairs, device,
            labelled_traj_weight=args.labelled_traj_weight)
        selection_loss = val_loss + args.delta_weight * val_delta
        message = (f"epoch {epoch + 1:3d}/{args.epochs} train={running / len(train_set):.4f} "
                   f"val={val_loss:.4f} delta={val_delta:.4f} "
                   f"intent_acc={intent_acc:.3f} ({time.time() - started:.0f}s)")

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            min_fde, mean_fde = displacement_error(
                model, val_loader, device, drop_plan=args.drop_plan)
            message += f" minFDE={min_fde:.2f}m meanFDE={mean_fde:.2f}m"

        print(message, flush=True)

        if selection_loss < best:
            best = selection_loss
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "config": {
                    "history_len": history_len, "future_len": future_len,
                    "n_neighbors": n_neighbors, "plan_len": plan_len,
                    "n_steps": args.n_steps,
                    "independent": bool(args.independent),
                    "token_conditioned": bool(args.token_conditioned),
                },
                "pos_scale": POS_SCALE, "vel_scale": VEL_SCALE,
                "resid_scale": RESID_SCALE,
                "drop_plan": bool(args.drop_plan),
                "independent": bool(args.independent),
                "type_weight": float(args.type_weight),
                "interventional_intent": bool(args.interventional_intent),
                "interventional_trajectory": bool(args.interventional_trajectory),
                "delta_weight": float(args.delta_weight),
                "labelled_traj_weight": float(args.labelled_traj_weight),
            }, args.out)

    print(f"best val loss {best:.4f}; saved to {args.out}")


if __name__ == "__main__":
    main()
