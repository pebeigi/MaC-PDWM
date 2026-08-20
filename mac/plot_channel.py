"""Plot ground-truth vs learned influence-channel probe response."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wm", default="data/mac/wm_eval.json")
    parser.add_argument("--out", default="paper/figures/channel.pdf")
    args = parser.parse_args()

    with open(args.wm) as handle:
        wm = json.load(handle)
    ch = wm["channel"]
    accels = np.asarray(ch["probe_accels"], dtype=float)
    names = {-4.0: "yield", -3.0: "yield", 0.0: "hold", 2.0: "assert", 3.0: "assert"}

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))

    ax = axes[0]
    ax.plot(accels, ch["p_yield_truth"], "o-", color="#444444",
            label="ground-truth kernel", lw=1.8)
    ax.plot(accels, ch["p_yield_model"], "s-", color="#1b6ca8",
            label="model (deciding)", lw=1.8)
    ax.plot(accels, ch.get("p_yield_model_all", [np.nan] * len(accels)),
            "^--", color="#d95f02", label="model (all present)", lw=1.5)
    ax.set_xticks(accels)
    ax.set_xticklabels([f"{names.get(a, a)}\n$a={a:+.0f}$" for a in accels])
    ax.set_ylabel(r"$\Pr(\mathrm{yield})$")
    ax.set_title("Channel recovery", fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, framealpha=0.9)

    ax = axes[1]
    # shifts[] are already vs the hold probe in wm_eval.json.
    shifts = [float(s) for s in ch["shifts"]]
    ax.bar([names.get(a, str(a)) for a in accels], shifts, color="#1b6ca8", alpha=0.85)
    ax.axhline(ch["spread"], color="#d95f02", ls="--", lw=1.5,
               label=f"sample spread ({ch['spread']:.2f} m)")
    ax.set_ylabel("trajectory shift vs hold (m)")
    ax.set_title("Predicted-response shift", fontsize=10)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.legend(fontsize=7.5, framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out} and {png}")


if __name__ == "__main__":
    main()
