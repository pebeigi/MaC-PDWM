"""Pick the intention-head guidance weight on held-out open-loop episodes.

The head reports the right sign for the plan effect but too small a magnitude.
Guidance rescales the plan contribution to the logits; the correct scale is a
property of the fit, not a free knob, so it is chosen here by minimising
validation cross-entropy on the interventional subset only -- the same subset
that identifies the causal conditional in the first place. The chosen weight is
written into the checkpoint so planners cannot pick their own.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from mac.models.diffusion_world_model import DiffusionWorldModel
from mac.train_world_model import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/mac/cross.npz")
    ap.add_argument("--checkpoint", default="data/mac/world_model_cross.pt")
    ap.add_argument("--weights", default="1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,8.0")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--write", action="store_true",
                    help="store the selected weight in the checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = DiffusionWorldModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    data = np.load(args.dataset)
    val = load_split(data, "val")
    history, ego_plan, _, types, interventional = val.tensors[:5]
    keep = interventional
    history, ego_plan, types = history[keep], ego_plan[keep], types[keep]
    print(f"{int(keep.sum())} open-loop validation samples, "
          f"{int((types >= 0).sum())} labelled neighbour slots")

    results = []
    for w in (float(x) for x in args.weights.split(",")):
        total, n, correct = 0.0, 0, 0
        with torch.no_grad():
            for i in range(0, len(history), args.batch_size):
                h = history[i:i + args.batch_size].to(device)
                u = ego_plan[i:i + args.batch_size].to(device)
                y = types[i:i + args.batch_size].to(device)
                valid = y >= 0
                if not valid.any():
                    continue
                probs = model.predict_intentions(h, u, guidance=w)
                logp = torch.log(probs.clamp_min(1e-8))[valid]
                yy = y[valid]
                total += float(F.nll_loss(logp, yy, reduction="sum"))
                correct += int((logp.argmax(dim=-1) == yy).sum())
                n += int(valid.sum())
        results.append((w, total / max(n, 1), correct / max(n, 1)))
        print(f"  guidance={w:4.1f}  val nll={results[-1][1]:.4f}  acc={results[-1][2]:.3f}")

    best = min(results, key=lambda r: r[1])
    print(f"\nselected guidance={best[0]} (nll={best[1]:.4f}, acc={best[2]:.3f})")
    if args.write:
        ckpt["guidance"] = float(best[0])
        torch.save(ckpt, args.checkpoint)
        print(f"wrote guidance={best[0]} into {args.checkpoint}")


if __name__ == "__main__":
    main()
