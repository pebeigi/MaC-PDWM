"""Six-frame time sequence of a single episode per map.

Each figure is one observation. Subplot times are linspace(t0, t_final, 6)
rounded to 0.1 s, where t0 / t_final are the first and last times the ego is
inside the plotted window (so every panel actually shows the ego).

Writes to a separate folder; does not overwrite paper/figures/.

    python -m mac.plot_setup_sequence --out paper/figures/time_sequence
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from mac.plot_setup import (
    TYPE_COLOR, EGO_COLOR, add_vehicle, draw_cross_roads, draw_merge_roads,
    draw_roundabout_roads, load_episode, type_legend,
)


def ego_times_in_view(record, xlim, ylim):
    """t0, t_final for frames where the ego is inside the axes window."""
    ego = record["ego_ids"][0]
    times = []
    for frame in record["frames"]:
        if ego not in frame["vehicles"]:
            continue
        x, y = frame["vehicles"][ego][:2]
        if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]:
            times.append(frame["t"])
    if not times:
        times = [record["frames"][0]["t"], record["frames"][-1]["t"]]
    return float(times[0]), float(times[-1])


def sequence_times(t0, t_final, n=6):
    raw = np.linspace(t0, t_final, n)
    return np.round(raw, 1)


def nearest_frame(record, t_target):
    ts = np.array([frame["t"] for frame in record["frames"]], dtype=float)
    return int(np.argmin(np.abs(ts - t_target)))


def plot_frame(ax, record, frame_idx, trail=True):
    ego = record["ego_ids"][0]
    types = record.get("driver_types", {})
    if trail:
        start = max(0, frame_idx - 10)
        for j in range(start, frame_idx, 2):
            fade = 0.10 + 0.30 * (j - start) / max(1, frame_idx - start)
            for vid, state in record["frames"][j]["vehicles"].items():
                x, y, _, _, _, heading, _ = state
                color = EGO_COLOR if vid == ego else TYPE_COLOR.get(
                    types.get(vid, ""), "#888888")
                add_vehicle(ax, x, y, heading, color, alpha=fade)
    frame = record["frames"][frame_idx]
    for vid, state in frame["vehicles"].items():
        x, y, _, _, _, heading, _ = state
        color = EGO_COLOR if vid == ego else TYPE_COLOR.get(
            types.get(vid, ""), "#888888")
        add_vehicle(ax, x, y, heading, color, alpha=1.0)
    t = frame["t"]
    ax.set_title(f"$t = {t:.1f}$ s", fontsize=11)


def plot_sequence(record, draw_roads, road_kwargs, xlim, ylim, outfile, title,
                  legend_loc="best"):
    t0, t_final = ego_times_in_view(record, xlim, ylim)
    times = sequence_times(t0, t_final, 6)
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 8.0))
    for ax, t in zip(axes.ravel(), times):
        draw_roads(ax, **road_kwargs)
        plot_frame(ax, record, nearest_frame(record, t))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    axes[0, 0].legend(handles=type_legend(), fontsize=7, loc=legend_loc,
                      frameon=True, fancybox=False, edgecolor="#ccc")
    outcome = record.get("outcome", "")
    fig.suptitle(
        f"{title}  ·  {outcome}  ·  "
        f"$t_0={t0:.1f}$ s  →  $t_\\mathrm{{final}}={t_final:.1f}$ s",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print("wrote", outfile, "times", list(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross", default="data/mac/raw_cross/ep_0001_00001.pkl")
    parser.add_argument("--merge", default="data/mac/raw_merge/ep_0001_00004.pkl")
    parser.add_argument("--roundabout",
                        default="data/mac/raw_roundabout/ep_0001_00002.pkl")
    parser.add_argument("--out", default="paper/figures/time_sequence")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    cross = load_episode(args.cross)
    plot_sequence(
        cross, draw_cross_roads, {"extent": 28},
        xlim=(-28, 28), ylim=(-28, 28),
        outfile=os.path.join(args.out, "setup_cross_trajectories.png"),
        title="Crossing — one episode")

    merge = load_episode(args.merge)
    plot_sequence(
        merge, draw_merge_roads, {"xlim": (-40, 18), "ylim": (-20, 10)},
        xlim=(-40, 18), ylim=(-20, 10),
        outfile=os.path.join(args.out, "setup_merge_trajectories.png"),
        title="Merge — one episode")

    roundabout = load_episode(args.roundabout)
    plot_sequence(
        roundabout, draw_roundabout_roads, {"extent": 42},
        xlim=(-42, 42), ylim=(-42, 42),
        outfile=os.path.join(args.out, "setup_roundabout.png"),
        title="Roundabout — one episode",
        legend_loc="upper right")


if __name__ == "__main__":
    main()
