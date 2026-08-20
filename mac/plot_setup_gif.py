"""Animated README overview: three maps + d(t), no side note.

Reuses the same drawing helpers as ``mac.plot_setup.write_overview``, but plays
each episode forward and writes a GIF.

    .venv-mac/bin/python -m mac.plot_setup_gif --out assets/setup_overview.gif
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from mac.plot_setup import (
    draw_cross_roads,
    draw_merge_roads,
    draw_roundabout_roads,
    load_episode,
    plot_snapshot,
    plot_timing,
    type_legend,
)
from mac.plot_setup_sequence import nearest_frame


def _ego_span(record, pad=2.0):
    """First/last times the ego is present, trimmed by ``pad`` seconds at each end."""
    ego = record["ego_ids"][0]
    times = [frame["t"] for frame in record["frames"] if ego in frame["vehicles"]]
    if not times:
        times = [record["frames"][0]["t"], record["frames"][-1]["t"]]
    t0 = float(times[0]) + pad
    t_final = float(times[-1]) - pad
    if t_final <= t0:
        t0, t_final = float(times[0]), float(times[-1])
    return t0, t_final


def _frame_indices(record, n_frames):
    """Evenly spaced frame indices over the ego's episode span."""
    t0, t_final = _ego_span(record)
    times = np.linspace(t0, t_final, n_frames)
    return [nearest_frame(record, float(t)) for t in times]


def _redraw_map(ax, draw_roads, road_kwargs, record, frame_idx, title, legend=False):
    ax.clear()
    draw_roads(ax, **road_kwargs)
    plot_snapshot(ax, record, frame_idx=frame_idx, trail=False, title=title)
    if legend:
        ax.legend(handles=type_legend(), fontsize=6, loc="upper left",
                  frameon=True, fancybox=False, edgecolor="#ccc")


def write_overview_gif(cross, merge, roundabout, outfile, n_frames=56, fps=5,
                       dpi=100):
    """2×3 overview (maps + timing) animated; no right-hand note column."""
    scenes = [
        {
            "record": cross,
            "draw": draw_cross_roads,
            "road_kwargs": {"extent": 20},
            "xlim": (-20, 20),
            "ylim": (-20, 20),
            "map_title": "Crossing",
            "time_title": "Crossing · d(t)",
        },
        {
            "record": merge,
            "draw": draw_merge_roads,
            "road_kwargs": {},
            "xlim": (-40, 18),
            "ylim": (-20, 10),
            "map_title": "Merge",
            "time_title": "Merge · d(t)",
        },
        {
            "record": roundabout,
            "draw": draw_roundabout_roads,
            "road_kwargs": {"extent": 40},
            "xlim": (-40, 40),
            "ylim": (-40, 40),
            "map_title": "Roundabout",
            "time_title": "Roundabout · d(t)",
        },
    ]

    indices = [_frame_indices(s["record"], n_frames) for s in scenes]

    fig = plt.figure(figsize=(11.4, 7.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0],
                          hspace=0.38, wspace=0.28)
    map_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    time_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    cursors = []

    for i, scene in enumerate(scenes):
        plot_timing(time_axes[i], scene["record"], title=scene["time_title"],
                    legend=(i == 0), ylabel=(i == 0), xlabel=True)
        line = time_axes[i].axvline(
            scene["record"]["frames"][indices[i][0]]["t"],
            color="#333333", lw=1.2, ls="-", zorder=6)
        cursors.append(line)

    fig.suptitle("MaC setup — one scene per map, true vehicle footprints",
                 fontsize=13, y=0.995)

    def update(k):
        artists = []
        for i, scene in enumerate(scenes):
            idx = indices[i][k]
            _redraw_map(
                map_axes[i], scene["draw"], scene["road_kwargs"],
                scene["record"], idx, scene["map_title"], legend=(i == 0))
            t = scene["record"]["frames"][idx]["t"]
            cursors[i].set_xdata([t, t])
            artists.append(cursors[i])
        return artists

    update(0)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps,
                         blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(outfile)) or ".", exist_ok=True)
    anim.save(outfile, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print("wrote", outfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross", default="data/mac/raw_cross/ep_0001_00001.pkl")
    parser.add_argument("--merge", default="data/mac/raw_merge/ep_0001_00004.pkl")
    parser.add_argument("--roundabout",
                        default="data/mac/raw_roundabout/ep_0001_00002.pkl")
    parser.add_argument("--out", default="assets/setup_overview.gif")
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    write_overview_gif(
        load_episode(args.cross),
        load_episode(args.merge),
        load_episode(args.roundabout),
        args.out,
        n_frames=args.frames,
        fps=args.fps,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
