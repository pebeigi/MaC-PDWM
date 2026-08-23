#!/usr/bin/env python3
"""Pre-flight validation for cross + roundabout scenarios."""
from __future__ import annotations

import os
import sys
import unittest
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

from mac.envs.sumo_planning_env import SCENARIOS, EnvConfig, SumoPlanningEnv
from mac.policies.rule_based import ConstantSpeedPolicy, TimeGapPolicy

PASS, FAIL, WARN = [], [], []


def ok(msg):
    PASS.append(msg)
    print(f"  OK  {msg}", flush=True)


def bad(msg):
    FAIL.append(msg)
    print(f" FAIL {msg}", flush=True)


def warn(msg):
    WARN.append(msg)
    print(f" WARN {msg}", flush=True)


def edges_in_net(net_path):
    root = ET.parse(net_path).getroot()
    return {
        e.get("id")
        for e in root.findall("edge")
        if e.get("id") and not e.get("id").startswith(":")
    }


def routes_in_rou(rou_path):
    root = ET.parse(rou_path).getroot()
    routes = {r.get("id"): r.get("edges", "").split() for r in root.findall("route")}
    vtypes = [v.get("id") for v in root.findall("vType")]
    return routes, vtypes


def rou_path_for(spec):
    return os.path.join(
        os.path.dirname(spec.sumocfg),
        os.path.basename(spec.sumocfg).replace(".sumocfg", ".rou.xml"),
    )


def check_scenario(name):
    print(f"\n======== {name} ========", flush=True)
    spec = SCENARIOS[name]
    rou = rou_path_for(spec)
    for path, label in (
        (spec.sumocfg, "sumocfg"),
        (spec.net_file, "net"),
        (rou, "rou"),
    ):
        if not os.path.isfile(path):
            bad(f"{name}: missing {label} {path}")
            return
    ok(f"{name}: files present")

    net_edges = edges_in_net(spec.net_file)
    routes, vtypes = routes_in_rou(rou)
    for vt in ("background", "ego"):
        if vt not in vtypes:
            bad(f"{name}: missing vType {vt}")
        else:
            ok(f"{name}: vType {vt}")

    for rid in spec.ego_routes:
        if rid not in routes:
            bad(f"{name}: ego route {rid} not in rou.xml")
            continue
        missing = [e for e in routes[rid] if e not in net_edges]
        if missing:
            bad(f"{name}: ego route {rid} missing edges {missing}")
        else:
            ok(f"{name}: ego route {rid} edges OK")

    for rid, rate in spec.background_routes.items():
        if rid not in routes:
            bad(f"{name}: background route {rid} not in rou.xml")
            continue
        missing = [e for e in routes[rid] if e not in net_edges]
        if missing:
            bad(f"{name}: bg route {rid} missing edges {missing}")
        elif rate <= 0:
            bad(f"{name}: bg route {rid} rate={rate}")
        else:
            ok(f"{name}: bg {rid} rate={rate} edges OK")

    rates = spec.background_routes
    total = sum(rates.values())
    oc, dc = Counter(), Counter()
    for r, rate in rates.items():
        oc[r[0]] += rate
        dc[r[1]] += rate
    print(f"  OD total={total:.3f} veh/s", flush=True)
    print(f"  origins={[(a, round(oc[a] / total, 3)) for a in sorted(oc)]}", flush=True)
    print(f"  dests  ={[(a, round(dc[a] / total, 3)) for a in sorted(dc)]}", flush=True)

    if name == "roundabout":
        for arm in "NEWS":
            if arm != "S" and oc[arm] / total < 0.05:
                warn(f"{name}: origin {arm} only {oc[arm] / total:.1%}")
            if dc[arm] / total < 0.05:
                warn(f"{name}: dest {arm} only {dc[arm] / total:.1%}")
        pass_south = {"WE", "NS", "WS", "ES", "NE", "WN"}
        share = sum(rates.get(r, 0) for r in pass_south) / total
        print(f"  share passing south merge: {share:.1%}", flush=True)
        if share < 0.25:
            warn(f"{name}: only {share:.1%} of bg passes south merge")
        elif share > 0.85:
            warn(f"{name}: {share:.1%} of bg passes south merge — may re-skew")
        else:
            ok(f"{name}: south-merge traffic share {share:.1%}")
        # SE/SW share S_in with ego
        s_share = (rates.get("SE", 0) + rates.get("SW", 0)) / total
        if s_share > 0.2:
            warn(f"{name}: S-origin bg share {s_share:.1%} may crowd ego entry")
        else:
            ok(f"{name}: light S-origin bg ({s_share:.1%})")

    if name == "cross":
        if set(rates) <= {"WE", "EW"}:
            ok(f"{name}: background is EW major only (by design for SN ego)")
        unused = set(routes) - set(rates) - set(spec.ego_routes)
        if unused:
            ok(f"{name}: unused rou ids (ok): {sorted(unused)}")


def live(name, episodes=5, horizon=60):
    outcomes = Counter()
    route_c = Counter()
    origin_c = Counter()
    dest_c = Counter()
    commits = 0
    reactives = 0
    ego_ok = 0
    errors = []
    for ep in range(episodes):
        cfg = EnvConfig(scenario=name, seed=100 + ep, horizon=horizon, warmup_steps=40)
        env = SumoPlanningEnv(cfg, label=f"chk_{name}_{ep}")
        try:
            env.reset(seed=100 + ep)
            real_add = env.conn.vehicle.add

            def counting_add(*a, **k):
                rid = k.get("routeID") if "routeID" in k else (a[1] if len(a) > 1 else None)
                vid = k.get("vehID") if "vehID" in k else (a[0] if a else "")
                if rid and not str(vid).startswith("ego"):
                    route_c[rid] += 1
                    origin_c[rid[0]] += 1
                    dest_c[rid[1]] += 1
                return real_add(*a, **k)

            env.conn.vehicle.add = counting_add
            for vid in env.conn.vehicle.getIDList():
                if vid.startswith("ego"):
                    ego_ok += 1
                    continue
                try:
                    rid = env.conn.vehicle.getRouteID(vid)
                    route_c[rid] += 1
                    origin_c[rid[0]] += 1
                    dest_c[rid[1]] += 1
                except Exception:
                    pass
            info = {"outcomes": [None]}
            for _ in range(horizon):
                _, _, done, info = env.step(np.array([0.0]))
                if done:
                    break
            outcomes[info["outcomes"][0] or "timeout"] += 1
            stats = info.get("driver_stats", {})
            commits += int(stats.get("reactive_resolved", 0) or 0)
            reactives += int(stats.get("reactive_opportunities", 0) or 0)
        except Exception as exc:
            errors.append(str(exc))
        finally:
            try:
                env.close()
            except Exception:
                pass
    return outcomes, route_c, origin_c, dest_c, commits, reactives, ego_ok, errors


def policy_smoke(name, episodes=8):
    rows = {}
    for pname, factory in (
        ("hold", lambda env: ConstantSpeedPolicy(env)),
        ("gap", lambda env: TimeGapPolicy(env)),
    ):
        outs = Counter()
        rets = []
        for ep in range(episodes):
            cfg = EnvConfig(scenario=name, seed=200 + ep, horizon=100, warmup_steps=40)
            env = SumoPlanningEnv(cfg, label=f"pol_{name}_{pname}_{ep}")
            try:
                pol = factory(env)
                obs = env.reset(seed=200 + ep)
                if hasattr(pol, "reset"):
                    pol.reset()
                total = 0.0
                info = {"outcomes": [None]}
                for _ in range(cfg.horizon):
                    obs, r, done, info = env.step(pol.act(obs))
                    total += float(np.sum(r))
                    if done:
                        break
                outs[info["outcomes"][0] or "timeout"] += 1
                rets.append(total)
            finally:
                env.close()
        rows[pname] = (dict(outs), float(np.mean(rets)))
    return rows


def main():
    print("Static checks", flush=True)
    check_scenario("cross")
    check_scenario("roundabout")

    print("\n======== LIVE SUMO ========", flush=True)
    for name in ("cross", "roundabout"):
        print(f"\n-- live {name} --", flush=True)
        outcomes, rc, oc, dc, commits, react, ego_ok, errors = live(name)
        for err in errors:
            bad(f"{name} live error: {err}")
        if not errors:
            ok(f"{name}: episodes completed without exception")
        print(f"  outcomes: {dict(outcomes)}", flush=True)
        print(f"  ego present resets: {ego_ok}", flush=True)
        total = sum(rc.values()) or 1
        print(f"  bg inserts counted: {sum(rc.values())}", flush=True)
        print(f"  origins: {[(a, round(oc[a] / total, 3)) for a in 'NEWS' if oc[a]]}", flush=True)
        print(f"  dests:   {[(a, round(dc[a] / total, 3)) for a in 'NEWS' if dc[a]]}", flush=True)
        print(f"  top routes: {rc.most_common(8)}", flush=True)
        print(f"  reactive commits/opportunities: {commits}/{react}", flush=True)
        if ego_ok < 5:
            bad(f"{name}: ego missing on many resets ({ego_ok}/5)")
        else:
            ok(f"{name}: ego inserted reliably")
        if name == "roundabout" and sum(rc.values()) > 0:
            if oc["E"] == 0 or dc["N"] == 0 or dc["W"] == 0:
                bad(f"{name} live: still missing an arm (E origin or N/W dest)")
            else:
                ok(f"{name} live: all major arms used")

    print("\n======== POLICY DIFFERENCE ========", flush=True)
    for name in ("cross", "roundabout"):
        rows = policy_smoke(name)
        print(f"  {name}:", flush=True)
        for p, (outs, ret) in rows.items():
            print(f"    {p:4s} return={ret:6.2f} outcomes={outs}", flush=True)
        if rows["hold"][0] == rows["gap"][0] and abs(rows["hold"][1] - rows["gap"][1]) < 0.5:
            warn(f"{name}: hold and gap look identical")
        else:
            ok(f"{name}: hold vs gap policies differ")

    print("\n======== UNIT TESTS ========", flush=True)
    suite = unittest.TestLoader().discover("tests")
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    if result.wasSuccessful():
        ok(f"unit tests passed ({result.testsRun})")
    else:
        bad(f"unit tests failed: failures={len(result.failures)} errors={len(result.errors)}")

    print("\n======== SUMMARY ========", flush=True)
    print(f"PASS {len(PASS)}  WARN {len(WARN)}  FAIL {len(FAIL)}", flush=True)
    for msg in WARN:
        print(" W", msg, flush=True)
    for msg in FAIL:
        print(" F", msg, flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
