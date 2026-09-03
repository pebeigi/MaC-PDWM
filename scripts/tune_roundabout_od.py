#!/usr/bin/env python3
"""Gate candidate roundabout OD configurations without training anything.

Three properties have to hold before a learned world model can be expected to
help, and none of them mention the world model:

  feasibility  a rule-based policy must sometimes clear the merge, otherwise
               every arm sits on the timeout/collision floor
  headroom     the non-causal branch oracle must score well above the rule-based
               policy, otherwise there is no action-dependent value to learn
  channel      reactive drivers must actually reach the decision zone while the
               ego is in play, otherwise influence cannot pay

Structural shares are reported alongside: background traffic on the ego's own
entry lane blocks insertion, and background traffic on the ego's downstream ring
edges decouples the merge outcome from the episode outcome.

    PYTHONPATH=. .venv-mac/bin/python scripts/tune_roundabout_od.py --stage fast
    PYTHONPATH=. .venv-mac/bin/python scripts/tune_roundabout_od.py --stage oracle --only clean_bal
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

from mac.envs.sumo_planning_env import SCENARIOS, EnvConfig, SumoPlanningEnv
from mac.eval_branch_oracle import run_branch_oracle, run_time_gap, summarize
from mac.models.belief import DEFAULT_PROBE_ACCELS

# Candidate demand tables (route id -> veh/s).
#
# Structural rules used to build them:
#   * nothing on S_in  - that is the ego's own insertion lane
#   * keep circ_WS well fed - that is the south-merge conflict the ego negotiates
#   * keep circ_EN light - the ego needs it to leave after winning the merge
CONFIGS: dict[str, dict[str, float]] = {
    # Reference: the pre-OD-fix design that produced the good oracle numbers.
    "orig": {"WE": 0.25, "NS": 0.20},

    # Current (regressed) design: S-origin traffic on the ego lane, heavy ring.
    "current": {
        "WE": 0.07, "EW": 0.07, "NS": 0.05,
        "WN": 0.05, "ES": 0.045, "NE": 0.045,
        "WS": 0.03, "EN": 0.045, "NW": 0.04,
        "SE": 0.025, "SW": 0.025,
    },

    # Balanced across arms, but ego lane clear and ego ring path clear.
    # Origins W+N, destinations E/S/W.
    "clean_bal": {
        "WE": 0.10, "WS": 0.06,
        "NS": 0.10, "NW": 0.06, "NE": 0.05,
    },

    # Same, lower demand (more natural gaps at the merge).
    "clean_bal_lite": {
        "WE": 0.08, "WS": 0.05,
        "NS": 0.08, "NW": 0.05, "NE": 0.04,
    },

    # Adds a light E origin and N destination for visual balance. EN/ES must
    # traverse circ_EN, so they stay small.
    "clean_bal_lightE": {
        "WE": 0.09, "WS": 0.05,
        "NS": 0.09, "NW": 0.05, "NE": 0.04,
        "EN": 0.03, "ES": 0.03,
    },

    # Conflict-heavy but ego path clear: more circ_WS feed, still no S_in.
    "merge_heavy": {
        "WE": 0.13, "WS": 0.07, "NS": 0.12, "NW": 0.05,
    },

    # All four arms visibly used, ego lane clear, circ_EN kept light. Demand
    # raised past clean_bal_lightE so the rule-based policy stops saturating.
    "bal_E_mid": {
        "WE": 0.10, "WS": 0.06,
        "NS": 0.10, "NW": 0.05, "NE": 0.045,
        "EN": 0.03, "ES": 0.03,
    },
    "bal_E_hard": {
        "WE": 0.12, "WS": 0.07,
        "NS": 0.12, "NW": 0.05, "NE": 0.05,
        "EN": 0.025, "ES": 0.025,
    },
}


def route_table():
    spec = SCENARIOS["roundabout"]
    rou = os.path.join(os.path.dirname(spec.sumocfg), "roundabout.rou.xml")
    root = ET.parse(rou).getroot()
    return {r.get("id"): r.get("edges").split() for r in root.findall("route")}


def structure(rates, routes):
    ego = routes["SN"]
    entry, ring = ego[0], set(ego[1:-1])
    total = sum(rates.values()) or 1.0
    on_entry = sum(r for rid, r in rates.items() if entry in routes[rid])
    on_ring = sum(r for rid, r in rates.items() if ring & set(routes[rid]))
    at_merge = sum(r for rid, r in rates.items() if "circ_WS" in routes[rid])
    origins = collections.Counter()
    dests = collections.Counter()
    for rid, rate in rates.items():
        origins[rid[0]] += rate
        dests[rid[1]] += rate
    return {
        "total": sum(rates.values()),
        "entry_share": on_entry / total,
        "ring_share": on_ring / total,
        "merge_share": at_merge / total,
        "origins": {a: round(origins[a] / total, 3) for a in "NEWS"},
        "dests": {a: round(dests[a] / total, 3) for a in "NEWS"},
    }


def make_env(name, tag, ep, horizon):
    cfg = EnvConfig(scenario="roundabout", seed=1000 + ep, horizon=horizon,
                    warmup_steps=40)
    return SumoPlanningEnv(cfg, label=f"od_{name}_{tag}_{ep}")


def run_gap(name, episodes, horizon):
    rows = []
    opps = []
    lost_at_spawn = 0
    for ep in range(episodes):
        env = make_env(name, "gap", ep, horizon)
        try:
            total, info = run_time_gap(env, seed=1000 + ep)
            outcome = info["outcomes"][0] or "timeout"
            rows.append({"return": total, "outcome": outcome})
            stats = info.get("driver_stats", {})
            opps.append(int(stats.get("reactive_opportunities",
                                      stats.get("reactive_resolved", 0)) or 0))
            if outcome == "lost":
                lost_at_spawn += 1
        finally:
            env.close()
    out = summarize(rows)
    out["lost_rate"] = lost_at_spawn / max(len(rows), 1)
    out["reactive_opportunities"] = float(np.mean(opps)) if opps else 0.0
    return out


def run_oracle(name, episodes, horizon, commit_steps=10):
    rows = []
    for ep in range(episodes):
        env = make_env(name, "orc", ep, horizon)
        try:
            total, info = run_branch_oracle(
                env, seed=1000 + ep, probes=list(DEFAULT_PROBE_ACCELS),
                commit_steps=commit_steps, exit_distance=30.0,
                rollout_steps=40)
            rows.append({"return": total,
                         "outcome": info["outcomes"][0] or "timeout"})
        finally:
            env.close()
    return summarize(rows)


def verdict(gap, orc):
    notes = []
    if gap["lost_rate"] > 0.02:
        notes.append(f"BLOCKED ego lost {gap['lost_rate']:.0%}")
    if gap["success_rate"] == 0.0 and gap["collision_rate"] > 0.8:
        notes.append("INFEASIBLE (gap policy always crashes)")
    if gap["timeout_rate"] > 0.7:
        notes.append("CONGESTED (gap policy mostly times out)")
    if gap["reactive_opportunities"] < 1.0:
        notes.append("NO CHANNEL (no reactive drivers in decision zone)")
    if orc is not None:
        if orc["success_rate"] < 0.6:
            notes.append(f"LOW HEADROOM (oracle {orc['success_rate']:.0%})")
        elif orc["return"] - gap["return"] < 5.0:
            notes.append("FLAT (oracle barely beats gap policy)")
        else:
            notes.append(f"OK headroom (+{orc['return'] - gap['return']:.1f} return)")
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fast", "oracle"], default="fast")
    ap.add_argument("--only", default="", help="comma-separated config names")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--oracle_episodes", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=150)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    routes = route_table()
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(CONFIGS)
    spec = SCENARIOS["roundabout"]
    saved = dict(spec.background_routes)
    results = {}

    try:
        for name in names:
            rates = CONFIGS[name]
            missing = [r for r in rates if r not in routes]
            if missing:
                print(f"{name}: SKIP (routes not in rou.xml: {missing})", flush=True)
                continue
            st = structure(rates, routes)
            print(f"\n===== {name} =====", flush=True)
            print(f"  demand={st['total']:.3f} veh/s   "
                  f"ego_entry={st['entry_share']:.0%}  "
                  f"ego_ring={st['ring_share']:.0%}  "
                  f"merge={st['merge_share']:.0%}", flush=True)
            print(f"  origins={st['origins']}  dests={st['dests']}", flush=True)

            spec.background_routes = dict(rates)
            gap = run_gap(name, args.episodes, args.horizon)
            print(f"  gap    : ret={gap['return']:7.2f} succ={gap['success_rate']:.2f} "
                  f"coll={gap['collision_rate']:.2f} tmo={gap['timeout_rate']:.2f} "
                  f"lost={gap['lost_rate']:.2f} opps={gap['reactive_opportunities']:.1f}",
                  flush=True)
            orc = None
            if args.stage == "oracle":
                orc = run_oracle(name, args.oracle_episodes, args.horizon)
                print(f"  oracle : ret={orc['return']:7.2f} succ={orc['success_rate']:.2f} "
                      f"coll={orc['collision_rate']:.2f} tmo={orc['timeout_rate']:.2f}",
                      flush=True)
            notes = verdict(gap, orc)
            print(f"  -> {'; '.join(notes) if notes else 'clean'}", flush=True)
            results[name] = {"structure": st, "gap": gap, "oracle": orc,
                             "notes": notes}
    finally:
        spec.background_routes = saved

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
