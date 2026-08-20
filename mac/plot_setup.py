"""Plot the SUMO maps and real episode trajectories.

Vehicles are drawn as 5 m × 1.8 m rectangles (SUMO vType size), zoomed to the
conflict. Timing plots are unchanged.

    python -m mac.plot_setup --out paper/figures
"""
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LANE = 3.2
VEH_LEN = 5.0
VEH_WID = 1.8
TYPE_COLOR = {
    "yielder": "#2ca02c",
    "contester": "#1f77b4",
    "reactive": "#9467bd",
}
EGO_COLOR = "#d62728"


def load_episode(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def tracks(record, radius=90.0):
    ego = record["ego_ids"][0]
    out = {}
    for frame in record["frames"]:
        t = frame["t"]
        for vid, state in frame["vehicles"].items():
            x, y, vx, vy, speed, heading, is_ego = state
            if abs(x) > radius + 40 or abs(y) > radius + 40:
                continue
            rec = out.setdefault(vid, {"x": [], "y": [], "t": [], "h": [], "ego": vid == ego})
            rec["x"].append(x)
            rec["y"].append(y)
            rec["t"].append(t)
            rec["h"].append(heading)
    kept = {}
    for vid, tr in out.items():
        dmin = min(np.hypot(np.asarray(tr["x"]), np.asarray(tr["y"])))
        if tr["ego"] or dmin < radius:
            kept[vid] = {k: (np.asarray(v) if k != "ego" else v) for k, v in tr.items()}
            kept[vid]["ego"] = tr["ego"]
    return kept, ego


def signed_conflict(xs, ys, conflict_point=(0.0, 0.0), headings=None):
    """Signed Euclidean distance to the conflict, matching the simulator.

    Positive = heading toward the point, negative = already passed it.
    Do not use radial distance to the origin: on the roundabout that is the
    ring radius (~22 m) and ``sign(diff(r))`` chatters every step.
    """
    cx, cy = conflict_point
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    approach_x = cx - xs
    approach_y = cy - ys
    distance = np.hypot(approach_x, approach_y)
    if headings is not None:
        headings = np.asarray(headings, dtype=float)
        toward = approach_x * np.cos(headings) + approach_y * np.sin(headings)
        return np.where(toward >= 0.0, distance, -distance)
    signed = distance.copy()
    if len(distance) >= 2:
        approaching = np.diff(distance, prepend=distance[0]) <= 0
        signed = np.where(approaching, distance, -distance)
    return signed


def conflict_point_for(record):
    from mac.envs.sumo_planning_env import SCENARIOS
    name = record.get("scenario", "cross")
    if name in SCENARIOS:
        return tuple(SCENARIOS[name].conflict_point)
    return (0.0, 0.0)


def vehicle_patch(x, y, heading, color, alpha=1.0, lw=0.7, z=4):
    """Axis-aligned in vehicle frame, then rotated. Position is vehicle centre."""
    c, s = np.cos(heading), np.sin(heading)
    hl, hw = VEH_LEN / 2.0, VEH_WID / 2.0
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    rot = np.array([[c, -s], [s, c]])
    world = local @ rot.T + np.array([x, y])
    body = mpatches.Polygon(world, closed=True, facecolor=color, edgecolor="#222",
                            linewidth=lw, alpha=alpha, zorder=z)
    # darker front bumper so heading is obvious
    front = np.array([[hl, hw * 0.7], [hl, -hw * 0.7],
                      [hl - 0.7, -hw * 0.7], [hl - 0.7, hw * 0.7]])
    front_w = front @ rot.T + np.array([x, y])
    nose = mpatches.Polygon(front_w, closed=True, facecolor="#111111",
                            edgecolor="none", alpha=min(1.0, alpha + 0.15), zorder=z + 0.1)
    return body, nose


def add_vehicle(ax, x, y, heading, color, alpha=1.0):
    body, nose = vehicle_patch(x, y, heading, color, alpha=alpha)
    ax.add_patch(body)
    ax.add_patch(nose)


def draw_cross_roads(ax, extent=30):
    w = LANE
    ax.fill_between([-extent, extent], -w, w, color="#d8d8d8", zorder=0)
    ax.fill_betweenx([-extent, extent], -w, w, color="#d8d8d8", zorder=0)
    edge = dict(color="#666666", lw=0.9, zorder=1)
    ax.plot([-extent, -w], [w, w], **edge)
    ax.plot([-extent, -w], [-w, -w], **edge)
    ax.plot([w, extent], [w, w], **edge)
    ax.plot([w, extent], [-w, -w], **edge)
    ax.plot([-w, -w], [-extent, -w], **edge)
    ax.plot([w, w], [-extent, -w], **edge)
    ax.plot([-w, -w], [w, extent], **edge)
    ax.plot([w, w], [w, extent], **edge)
    ax.plot([-extent, extent], [0, 0], color="#b0b0b0", lw=0.5, ls=":", zorder=1)
    ax.plot([0, 0], [-extent, extent], color="#b0b0b0", lw=0.5, ls=":", zorder=1)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def _offset_polyline(points, dist):
    """Shift a polyline left of its travel direction by ``dist`` metres."""
    pts = np.asarray(points, dtype=float)
    tangents = np.zeros_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True).clip(min=1e-9)
    tangents = tangents / norms
    left = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    return pts + dist * left


def _densify(points, step=4.0):
    pts = np.asarray(points, dtype=float)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return np.asarray(out)


def draw_merge_roads(ax, xlim=(-40, 18), ylim=(-20, 10)):
    """Zipper merge: ramp pavement joins the mainline.

    Solid outer edges. Dashed lane line between ramp and main through the
    zipper; after the merge the remaining lane has solid edges only.
    """
    w = LANE
    north, south = 0.0, -w
    zip_start, zip_end = -17.50, 1.82
    x0, x1 = xlim[0] - 12.0, xlim[1] + 12.0
    ramp_center = _densify([
        [-199.69, -41.57],
        [-17.16, -5.06],
        [-12.83, -4.11],
        [-7.71, -2.96],
        [-2.57, -2.00],
        [zip_end, -1.60],
    ])
    ramp_inner = _offset_polyline(ramp_center, w / 2.0)
    ramp_outer = _offset_polyline(ramp_center, -w / 2.0)

    ax.fill([x0, x1, x1, x0], [north, north, south, south],
            color="#d8d8d8", zorder=0)
    ax.fill(np.r_[ramp_inner[:, 0], ramp_outer[::-1, 0]],
            np.r_[ramp_inner[:, 1], ramp_outer[::-1, 1]],
            color="#d8d8d8", zorder=0)

    edge = dict(color="#666666", lw=0.95, zorder=2, solid_capstyle="butt")
    dash = dict(color="#777777", lw=0.9, zorder=2, ls=(0, (3.0, 2.2)),
                solid_capstyle="butt")
    ax.plot([x0, x1], [north, north], **edge)
    ax.plot([x0, zip_start], [south, south], **edge)
    ax.plot([zip_start, zip_end], [south, south], **dash)
    ax.plot([zip_end, x1], [south, south], **edge)
    ax.plot(ramp_outer[:, 0], ramp_outer[:, 1], **edge)
    pre = ramp_inner[ramp_inner[:, 0] <= zip_start]
    if len(pre):
        ax.plot(pre[:, 0], pre[:, 1], **edge)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def draw_roundabout_roads(ax, radius=20.0, extent=45):
    """Single-lane 4-arm roundabout used by the third scenario."""
    w = LANE
    R = float(radius)
    ax.add_patch(plt.Circle((0, 0), R + w, color="#d8d8d8", zorder=0))
    ax.add_patch(plt.Circle((0, 0), max(R - w, 0.5), color="white", zorder=1))
    circ = dict(color="#666666", lw=0.9, zorder=2, fill=False)
    ax.add_patch(plt.Circle((0, 0), R + w, **circ))
    ax.add_patch(plt.Circle((0, 0), max(R - w, 0.5), **circ))
    ax.fill_between([-w, w], -extent, -(R + w), color="#d8d8d8", zorder=0)
    ax.fill_between([-w, w], R + w, extent, color="#d8d8d8", zorder=0)
    ax.fill_betweenx([-w, w], -extent, -(R + w), color="#d8d8d8", zorder=0)
    ax.fill_betweenx([-w, w], R + w, extent, color="#d8d8d8", zorder=0)
    edge = dict(color="#666666", lw=0.9, zorder=2)
    ax.plot([-w, -w], [-extent, -(R + w)], **edge)
    ax.plot([w, w], [-extent, -(R + w)], **edge)
    ax.plot([-w, -w], [R + w, extent], **edge)
    ax.plot([w, w], [R + w, extent], **edge)
    ax.plot([-extent, -(R + w)], [-w, -w], **edge)
    ax.plot([-extent, -(R + w)], [w, w], **edge)
    ax.plot([R + w, extent], [-w, -w], **edge)
    ax.plot([R + w, extent], [w, w], **edge)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def ego_signed_in_frame(frame, ego_id):
    if ego_id not in frame["vehicles"]:
        return None
    x, y = frame["vehicles"][ego_id][:2]
    return float(np.hypot(x, y))


def pick_frame_index(record, target_dist=18.0):
    """Frame where ego is approaching and closest to ``target_dist`` from origin."""
    ego = record["ego_ids"][0]
    best_i, best_gap = 0, 1e9
    prev = None
    for i, frame in enumerate(record["frames"]):
        if ego not in frame["vehicles"]:
            continue
        x, y = frame["vehicles"][ego][:2]
        d = float(np.hypot(x, y))
        approaching = True if prev is None else d <= prev + 0.2
        prev = d
        if not approaching:
            continue
        gap = abs(d - target_dist)
        if gap < best_gap:
            best_i, best_gap = i, gap
    # collision / timeout: also consider last frame if ego never got that close
    if best_gap > 12 and record["frames"]:
        return len(record["frames"]) - 1
    return best_i


def plot_snapshot(ax, record, frame_idx=None, target_dist=18.0, trail=True,
                  trail_every=2, title=""):
    """Zoomed snapshot: rectangles at one time, faint earlier poses as trail."""
    ego = record["ego_ids"][0]
    types = record.get("driver_types", {})
    if frame_idx is None:
        frame_idx = pick_frame_index(record, target_dist=target_dist)
    frame = record["frames"][frame_idx]
    t = frame["t"]

    if trail:
        start = max(0, frame_idx - 12)
        for j in range(start, frame_idx, trail_every):
            fr = record["frames"][j]
            fade = 0.12 + 0.35 * (j - start) / max(1, frame_idx - start)
            for vid, state in fr["vehicles"].items():
                x, y, _, _, _, heading, _ = state
                color = EGO_COLOR if vid == ego else TYPE_COLOR.get(types.get(vid, ""), "#888888")
                add_vehicle(ax, x, y, heading, color, alpha=fade)

    for vid, state in frame["vehicles"].items():
        x, y, _, _, _, heading, _ = state
        color = EGO_COLOR if vid == ego else TYPE_COLOR.get(types.get(vid, ""), "#888888")
        add_vehicle(ax, x, y, heading, color, alpha=1.0)

    if ego in frame["vehicles"]:
        ex, ey = frame["vehicles"][ego][:2]
        ax.annotate("ego", (ex + 1.2, ey + 2.2), color=EGO_COLOR, fontsize=8,
                    fontweight="bold", zorder=8)

    outcome = record.get("outcome", "")
    ax.set_title(f"{title}\n$t={t:.1f}$ s · {outcome}", fontsize=10)


def plot_timing(ax, record, title="", legend=True, ylabel=True, xlabel=True):
    trs, ego = tracks(record, radius=120)
    types = record.get("driver_types", {})
    point = conflict_point_for(record)
    ego_tr = next(tr for tr in trs.values() if tr["ego"])
    ax.plot(ego_tr["t"], signed_conflict(
                ego_tr["x"], ego_tr["y"], point, headings=ego_tr.get("h")),
            color=EGO_COLOR, lw=2.4, label="ego", zorder=5)
    others = []
    for vid, tr in trs.items():
        if tr["ego"] or len(tr["x"]) < 4:
            continue
        span_x = tr["x"].max() - tr["x"].min()
        dmin = np.hypot(tr["x"] - point[0], tr["y"] - point[1]).min()
        if span_x > 20 and dmin < 60:
            others.append((dmin, vid, tr))
    others.sort()
    seen = set()
    for _, vid, tr in others[:4]:
        kind = types.get(vid, "?")
        label = kind if kind not in seen else None
        seen.add(kind)
        ax.plot(tr["t"], signed_conflict(
                    tr["x"], tr["y"], point, headings=tr.get("h")),
                color=TYPE_COLOR.get(kind, "#7f7f7f"), lw=1.4, alpha=0.9,
                label=label)
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    if xlabel:
        ax.set_xlabel("time (s)")
    if ylabel:
        ax.set_ylabel("signed dist. to conflict (m)")
    if title:
        ax.set_title(title, fontsize=12)
    if legend:
        ax.legend(fontsize=8, loc="upper right", frameon=False, handlelength=1.2)
    ax.set_ylim(-80, 120)
    ax.set_xlim(left=0)


def type_legend():
    return [
        mpatches.Patch(facecolor=EGO_COLOR, edgecolor="#222", label="ego  5×1.8 m"),
        mpatches.Patch(facecolor=TYPE_COLOR["yielder"], edgecolor="#222",
                       label="yielder (always concede)"),
        mpatches.Patch(facecolor=TYPE_COLOR["contester"], edgecolor="#222",
                       label="contester (never concede)"),
        mpatches.Patch(facecolor=TYPE_COLOR["reactive"], edgecolor="#222",
                       label="reactive (β₂ · ego accel)"),
    ]


def write_timing_samples(rows, outfile):
    """Big d(t) grid: each row is one episode sample, columns are the three maps.

    ``rows`` is a list of ``(cross_record, merge_record, roundabout_record)``.
    """
    from matplotlib.lines import Line2D

    n = len(rows)
    row_h = 1.95 if n >= 6 else 3.15
    fig, axes = plt.subplots(n, 3, figsize=(12.6, row_h * n + 0.85),
                             sharex=False, sharey=True)
    if n == 1:
        axes = np.array([axes])
    col_title = ("Crossing", "Merge", "Roundabout")
    mid = n // 2
    for r, triple in enumerate(rows):
        for c, record in enumerate(triple):
            ax = axes[r, c]
            plot_timing(ax, record, title="", legend=False,
                        ylabel=False, xlabel=False)
            ax.tick_params(labelsize=7, length=2.5)
            if r == 0:
                ax.set_title(col_title[c], fontsize=12, pad=4)
            if r != n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("time (s)", fontsize=10)
            if c == 0:
                ax.annotate(f"{r + 1}", xy=(-0.16, 0.5), xycoords="axes fraction",
                            ha="right", va="center", fontsize=9, fontweight="bold")
                if r == mid:
                    ax.set_ylabel("signed dist. to conflict (m)", fontsize=10)
            ax.set_xlim(0, 40)
    handles = [
        Line2D([0], [0], color=EGO_COLOR, lw=2.2, label="ego"),
        Line2D([0], [0], color=TYPE_COLOR["yielder"], lw=1.3, label="yielder"),
        Line2D([0], [0], color=TYPE_COLOR["contester"], lw=1.3, label="contester"),
        Line2D([0], [0], color=TYPE_COLOR["reactive"], lw=1.3, label="reactive"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(h_pad=2.7, w_pad=0.4)
    os.makedirs(os.path.dirname(os.path.abspath(outfile)) or ".", exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    pdf = os.path.splitext(outfile)[0] + ".pdf"
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", outfile)
    print("wrote", pdf)


def write_overview(cross, merge, roundabout, outfile, note=None):
    """One snapshot + d(t) per map, with a side note. Does not close the figure."""
    fig = plt.figure(figsize=(14.2, 7.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 0.85],
                          height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    draw_cross_roads(ax, extent=20)
    plot_snapshot(ax, cross, target_dist=14, title="Crossing")
    ax.legend(handles=type_legend(), fontsize=6, loc="upper left",
              frameon=True, fancybox=False, edgecolor="#ccc")

    ax = fig.add_subplot(gs[0, 1])
    draw_merge_roads(ax)
    plot_snapshot(ax, merge, target_dist=16, title="Merge")

    ax = fig.add_subplot(gs[0, 2])
    draw_roundabout_roads(ax, extent=40)
    plot_snapshot(ax, roundabout, target_dist=18, title="Roundabout")

    ax = fig.add_subplot(gs[1, 0])
    plot_timing(ax, cross, title="Crossing · d(t)")
    ax = fig.add_subplot(gs[1, 1])
    plot_timing(ax, merge, title="Merge · d(t)")
    ax = fig.add_subplot(gs[1, 2])
    plot_timing(ax, roundabout, title="Roundabout · d(t)")

    ax = fig.add_subplot(gs[:, 3])
    ax.axis("off")
    if note is None:
        note = (
            "What is being tested\n\n"
            "Task  Ego clears an unprotected\n"
            "conflict; SUMO ROW is off.\n\n"
            "Crossing  Ego S→N, bg E↔W.\n"
            "Merge     Ego ramp → mainline.\n"
            "Roundabout  4-arm, south entry.\n\n"
            "Cars  5.0 m × 1.8 m (SUMO).\n"
            "Dark strip = front bumper.\n\n"
            "Hidden types (not in obs)\n"
            "  yielder    always concede\n"
            "  contester  never concede\n"
            "  reactive   σ(β₁ΔTTC+β₂ȧ)\n\n"
            "β₂=2.5, D=35 m (RA 22 m).\n"
            "0.8 s EMA, Δ=0.4 s.\n"
            "a∈{−4,−2,0,1,2,3}."
        )
    ax.text(0.02, 0.98, note, va="top", ha="left", fontsize=8,
            family="DejaVu Sans Mono", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.45", fc="#f7f7f7", ec="#cccccc"))
    fig.suptitle("MaC setup — one scene per map, true vehicle footprints",
                 fontsize=13, y=0.995)
    os.makedirs(os.path.dirname(os.path.abspath(outfile)) or ".", exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print("wrote", outfile)


def _episode_score(record, target_dist, view_extent):
    """Prefer successful, busy approaches with a real conflict crossing."""
    if record.get("outcome") != "success":
        return -1.0
    ego = record["ego_ids"][0]
    idx = pick_frame_index(record, target_dist=target_dist)
    frame = record["frames"][idx]
    if ego not in frame["vehicles"]:
        return -1.0
    types = record.get("driver_types", {})
    n_view = 0
    kinds = set()
    for vid, state in frame["vehicles"].items():
        x, y = state[0], state[1]
        if abs(x) > view_extent or abs(y) > view_extent:
            continue
        n_view += 1
        if vid != ego:
            kinds.add(types.get(vid, "?"))
    if n_view < 2:
        return -1.0
    point = conflict_point_for(record)
    xs, ys, hs = [], [], []
    for fr in record["frames"]:
        if ego not in fr["vehicles"]:
            continue
        x, y, _, _, _, heading, _ = fr["vehicles"][ego]
        xs.append(x)
        ys.append(y)
        hs.append(heading)
    signed = signed_conflict(xs, ys, point, headings=hs)
    if signed.max() < 8.0 or signed.min() > -2.0:
        return -1.0
    return float(n_view + 4 * len(kinds) + 0.02 * len(record["frames"]))


def rank_episodes(directory, target_dist, view_extent, limit=120, want=25):
    ranked = []
    names = sorted(f for f in os.listdir(directory) if f.endswith(".pkl"))[:limit]
    for name in names:
        path = os.path.join(directory, name)
        try:
            record = load_episode(path)
        except Exception:
            continue
        score = _episode_score(record, target_dist, view_extent)
        if score < 0:
            continue
        ranked.append((score, name, path, record.get("outcome", "")))
        if want and len(ranked) >= want:
            break
    ranked.sort(reverse=True)
    return ranked
    return [
        mpatches.Patch(facecolor=EGO_COLOR, edgecolor="#222", label="ego  5×1.8 m"),
        mpatches.Patch(facecolor=TYPE_COLOR["yielder"], edgecolor="#222",
                       label="yielder (always concede)"),
        mpatches.Patch(facecolor=TYPE_COLOR["contester"], edgecolor="#222",
                       label="contester (never concede)"),
        mpatches.Patch(facecolor=TYPE_COLOR["reactive"], edgecolor="#222",
                       label="reactive (β₂ · ego accel)"),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross_dir", default="data/mac/raw_cross")
    parser.add_argument("--merge_dir", default="data/mac/raw_merge")
    parser.add_argument("--roundabout_dir", default="data/mac/raw_roundabout")
    parser.add_argument("--out", default="paper/figures")
    parser.add_argument("--candidates", type=int, default=0,
                        help="If >0, write this many overview samples and exit.")
    parser.add_argument("--candidate_dir",
                        default="paper/figures/overview_candidates")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "figure.dpi": 140,
        "savefig.dpi": 170,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    if args.candidates > 0:
        os.makedirs(args.candidate_dir, exist_ok=True)
        cross_r = rank_episodes(args.cross_dir, 14, 20, limit=400, want=20)
        merge_r = rank_episodes(args.merge_dir, 16, 40, limit=800, want=20)
        rbt_r = rank_episodes(args.roundabout_dir, 18, 40, limit=400, want=20)
        n = min(len(cross_r), len(merge_r), len(rbt_r))
        if n == 0:
            raise SystemExit("no successful busy episodes found")
        lines = ["# overview candidates — copy one to paper/figures/setup_overview.png",
                 "# ranked from first 150 pickles/map, success + busy snapshot", ""]
        combos = []
        seen = set()
        i = 0
        while len(combos) < args.candidates and i < n * 4:
            triple = (i % n, (2 * i) % n, (3 * i + 1) % n)
            if triple not in seen:
                seen.add(triple)
                combos.append(triple)
            i += 1
        for k, (ic, im, ir) in enumerate(combos, 1):
            c_name, c_path = cross_r[ic][1], cross_r[ic][2]
            m_name, m_path = merge_r[im][1], merge_r[im][2]
            r_name, r_path = rbt_r[ir][1], rbt_r[ir][2]
            outfile = os.path.join(args.candidate_dir, f"setup_overview_{k:02d}.png")
            write_overview(load_episode(c_path), load_episode(m_path),
                           load_episode(r_path), outfile)
            lines.append(
                f"{k:02d}  cross={c_name}  merge={m_name}  roundabout={r_name}")
        manifest = os.path.join(args.candidate_dir, "README.txt")
        with open(manifest, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        print("wrote", manifest)
        return

    cross_success = load_episode(os.path.join(args.cross_dir, "ep_0001_00006.pkl"))
    cross_collide = load_episode(os.path.join(args.cross_dir, "ep_0001_00000.pkl"))
    cross_busy = load_episode(os.path.join(args.cross_dir, "ep_0001_00001.pkl"))
    merge_success = load_episode(os.path.join(args.merge_dir, "ep_0001_00002.pkl"))
    merge_collide = load_episode(os.path.join(args.merge_dir, "ep_0001_00001.pkl"))
    rbt_success = load_episode(os.path.join(args.roundabout_dir, "ep_0001_00002.pkl"))
    rbt_collide = load_episode(os.path.join(args.roundabout_dir, "ep_0001_00000.pkl"))

    # --- crossing: 3 times × 2 outcomes, zoomed ±28 m ---
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 8.2))
    times_cross = (22.0, 14.0, 8.0)
    extents_cross = (26, 20, 16)
    labels = ("~22 m out", "~14 m out", "~8 m / junction")
    for col, (dist, ext, lab) in enumerate(zip(times_cross, extents_cross, labels)):
        draw_cross_roads(axes[0, col], extent=ext)
        plot_snapshot(axes[0, col], cross_success, target_dist=dist,
                      title=f"Success · {lab}")
        draw_cross_roads(axes[1, col], extent=ext)
        plot_snapshot(axes[1, col], cross_collide, target_dist=dist,
                      title=f"Collision · {lab}")
    axes[0, 0].legend(handles=type_legend(), fontsize=7, loc="upper left",
                      frameon=True, fancybox=False, edgecolor="#ccc")
    fig.suptitle("Crossing — 5 m × 1.8 m cars, progressively zoomed to the junction",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.out, "setup_cross_trajectories.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)

    # --- merge: zoomed around zipper ---
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.4))
    times_merge = (35.0, 18.0, 8.0)
    labels = ("~35 m out", "~18 m out", "~8 m / zipper")
    for col, (dist, lab) in enumerate(zip(times_merge, labels)):
        draw_merge_roads(axes[0, col])
        plot_snapshot(axes[0, col], merge_success, target_dist=dist,
                      title=f"Success · {lab}")
        draw_merge_roads(axes[1, col])
        plot_snapshot(axes[1, col], merge_collide, target_dist=dist,
                      title=f"Collision · {lab}")
    axes[0, 0].legend(handles=type_legend(), fontsize=7, loc="lower left",
                      frameon=True, fancybox=False, edgecolor="#ccc")
    fig.suptitle("Merge — same vehicle size, zoomed to the zipper", fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.out, "setup_merge_trajectories.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)

    # --- overview: one snapshot + timing per scenario, note on the right ---
    write_overview(cross_busy, merge_success, rbt_success,
                   os.path.join(args.out, "setup_overview.png"))

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4))
    plot_timing(axes[0], cross_busy, title="Crossing · d(t) to conflict")
    plot_timing(axes[1], merge_success, title="Merge · d(t) to conflict")
    plot_timing(axes[2], rbt_success, title="Roundabout · d(t) to conflict")
    fig.tight_layout()
    out = os.path.join(args.out, "setup_timing.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    draw_roundabout_roads(axes[0])
    plot_snapshot(axes[0], rbt_success, target_dist=18,
                  title="Roundabout (success)")
    draw_roundabout_roads(axes[1])
    plot_snapshot(axes[1], rbt_collide, target_dist=18,
                  title="Roundabout (collision)")
    axes[0].legend(handles=type_legend(), fontsize=7, loc="upper right",
                   frameon=True, fancybox=False, edgecolor="#ccc")
    fig.suptitle("Roundabout — same vehicle size, conflict at the south entry",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.out, "setup_roundabout.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
