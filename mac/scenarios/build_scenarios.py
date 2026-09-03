"""Generate the SUMO networks and demand files used by the MaC planning envs.

Each scenario is written as plain node/edge XML and compiled with netconvert so
the junction control mode (priority vs. unregulated) stays explicit and
reproducible instead of being inherited from a pre-built network.
"""
import argparse
import math
import os
import subprocess
import sys

SCENARIO_DIR = os.path.dirname(os.path.abspath(__file__))

# Arm length is long enough for a vehicle to accelerate to cruise speed and
# still leave ~4 s of approach time for negotiation to be observable.
ARM_LENGTH = 150.0


def _write(path, text):
    with open(path, "w") as handle:
        handle.write(text.strip() + "\n")


def _netconvert(node_file, edge_file, out_file, extra_args=None):
    cmd = [
        "netconvert",
        "--node-files", node_file,
        "--edge-files", edge_file,
        "--output-file", out_file,
        "--no-turnarounds", "true",
        "--offset.disable-normalization", "true",
    ]
    if extra_args:
        cmd += extra_args
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def build_unsignalized_cross(out_dir, junction_type="priority"):
    """Four-arm unsignalized crossing.

    East-West is the major road, North-South the minor road. With
    ``junction_type='priority'`` SUMO enforces right-of-way for background
    traffic, while ego vehicles (whose safety checks are disabled by the env)
    must negotiate a gap themselves.
    """
    os.makedirs(out_dir, exist_ok=True)
    L = ARM_LENGTH

    nodes = f"""
<nodes>
    <node id="center" x="0.0" y="0.0" type="{junction_type}"/>
    <node id="west"   x="{-L}" y="0.0" type="priority"/>
    <node id="east"   x="{L}"  y="0.0" type="priority"/>
    <node id="south"  x="0.0" y="{-L}" type="priority"/>
    <node id="north"  x="0.0" y="{L}"  type="priority"/>
</nodes>
"""

    # priority 2 on the major road makes the conflict asymmetric, which is what
    # forces the minor-road agent to communicate intent through motion.
    edges = """
<edges>
    <edge id="W_in"  from="west"   to="center" numLanes="1" speed="13.89" priority="2"/>
    <edge id="E_out" from="center" to="east"   numLanes="1" speed="13.89" priority="2"/>
    <edge id="E_in"  from="east"   to="center" numLanes="1" speed="13.89" priority="2"/>
    <edge id="W_out" from="center" to="west"   numLanes="1" speed="13.89" priority="2"/>
    <edge id="S_in"  from="south"  to="center" numLanes="1" speed="13.89" priority="1"/>
    <edge id="N_out" from="center" to="north"  numLanes="1" speed="13.89" priority="1"/>
    <edge id="N_in"  from="north"  to="center" numLanes="1" speed="13.89" priority="1"/>
    <edge id="S_out" from="center" to="south"  numLanes="1" speed="13.89" priority="1"/>
</edges>
"""

    node_file = os.path.join(out_dir, "cross.nod.xml")
    edge_file = os.path.join(out_dir, "cross.edg.xml")
    net_file = os.path.join(out_dir, "cross.net.xml")
    _write(node_file, nodes)
    _write(edge_file, edges)
    _netconvert(node_file, edge_file, net_file)

    routes = """
<routes>
    <!-- A finite reaction time (actionStepLength) and a bounded emergency decel
         are what make a badly timed ego manoeuvre genuinely unrecoverable. -->
    <vType id="background" accel="2.6" decel="4.5" emergencyDecel="6.0" sigma="0.5"
           length="5.0" minGap="2.5" maxSpeed="13.89" tau="1.0" carFollowModel="IDM"
           actionStepLength="0.8" speedFactor="normc(1.0,0.1,0.8,1.2)"/>
    <!-- Contesting drivers keep their right of way: they do not brake for a
         vehicle that cuts into the conflict zone ahead of them. -->
    <vType id="bg_contest" accel="2.6" decel="4.5" emergencyDecel="4.5" sigma="0.5"
           length="5.0" minGap="2.5" maxSpeed="13.89" tau="0.9" carFollowModel="IDM"
           actionStepLength="1.0" speedFactor="normc(1.05,0.1,0.85,1.25)"
           jmIgnoreFoeProb="1.0" jmIgnoreFoeSpeed="30.0" jmTimegapMinor="0"
           jmCrossingGap="0" color="0.2,0.4,1"/>
    <vType id="ego" accel="3.0" decel="6.0" emergencyDecel="9.0" sigma="0.0"
           length="5.0" minGap="2.0" maxSpeed="13.89" tau="1.0" carFollowModel="IDM"
           color="1,0.4,0"/>

    <route id="WE" edges="W_in E_out"/>
    <route id="EW" edges="E_in W_out"/>
    <route id="SN" edges="S_in N_out"/>
    <route id="NS" edges="N_in S_out"/>
    <route id="SW" edges="S_in W_out"/>
    <route id="WN" edges="W_in N_out"/>
</routes>
"""
    _write(os.path.join(out_dir, "cross.rou.xml"), routes)

    sumocfg = """
<configuration>
    <input>
        <net-file value="cross.net.xml"/>
        <route-files value="cross.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <step-length value="0.2"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
        <collision.action value="warn"/>
        <collision.check-junctions value="true"/>
        <collision.mingap-factor value="0"/>
        <default.speeddev value="0.1"/>
    </processing>
    <report>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
</configuration>
"""
    _write(os.path.join(out_dir, "cross.sumocfg"), sumocfg)
    return net_file


def build_merge(out_dir):
    """Two-lane-to-one merge: the ego must interleave with mainline traffic."""
    os.makedirs(out_dir, exist_ok=True)

    nodes = """
<nodes>
    <node id="main_start" x="-300.0" y="0.0"  type="priority"/>
    <node id="merge_pt"   x="0.0"    y="0.0"  type="zipper"/>
    <node id="main_end"   x="300.0"  y="0.0"  type="priority"/>
    <node id="ramp_start" x="-200.0" y="-40.0" type="priority"/>
</nodes>
"""

    edges = """
<edges>
    <edge id="main_in"  from="main_start" to="merge_pt" numLanes="1" speed="22.22" priority="2"/>
    <edge id="ramp_in"  from="ramp_start" to="merge_pt" numLanes="1" speed="16.67" priority="1"/>
    <edge id="main_out" from="merge_pt"   to="main_end" numLanes="1" speed="22.22" priority="2"/>
</edges>
"""

    node_file = os.path.join(out_dir, "merge.nod.xml")
    edge_file = os.path.join(out_dir, "merge.edg.xml")
    net_file = os.path.join(out_dir, "merge.net.xml")
    _write(node_file, nodes)
    _write(edge_file, edges)
    _netconvert(node_file, edge_file, net_file)

    routes = """
<routes>
    <vType id="background" accel="2.6" decel="4.5" emergencyDecel="6.0" sigma="0.5"
           length="5.0" minGap="2.5" maxSpeed="22.22" tau="1.0" carFollowModel="IDM"
           actionStepLength="0.8" speedFactor="normc(1.0,0.1,0.8,1.2)"/>
    <vType id="bg_contest" accel="2.6" decel="4.5" emergencyDecel="4.5" sigma="0.5"
           length="5.0" minGap="2.5" maxSpeed="22.22" tau="0.9" carFollowModel="IDM"
           actionStepLength="1.0" speedFactor="normc(1.05,0.1,0.85,1.25)"
           jmIgnoreFoeProb="1.0" jmIgnoreFoeSpeed="30.0" jmTimegapMinor="0"
           jmCrossingGap="0" color="0.2,0.4,1"/>
    <vType id="ego" accel="3.0" decel="6.0" emergencyDecel="9.0" sigma="0.0"
           length="5.0" minGap="2.0" maxSpeed="22.22" tau="1.0" carFollowModel="IDM"
           color="1,0.4,0"/>

    <route id="MAIN" edges="main_in main_out"/>
    <route id="RAMP" edges="ramp_in main_out"/>
</routes>
"""
    _write(os.path.join(out_dir, "merge.rou.xml"), routes)

    sumocfg = """
<configuration>
    <input>
        <net-file value="merge.net.xml"/>
        <route-files value="merge.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <step-length value="0.2"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
        <collision.action value="warn"/>
        <collision.check-junctions value="true"/>
        <collision.mingap-factor value="0"/>
        <default.speeddev value="0.1"/>
    </processing>
    <report>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
</configuration>
"""
    _write(os.path.join(out_dir, "merge.sumocfg"), sumocfg)
    return net_file


def _arc_shape(radius, deg0, deg1, n=10):
    pts = []
    for i in range(n + 1):
        ang = math.radians(deg0 + (deg1 - deg0) * i / n)
        pts.append(f"{radius * math.cos(ang):.3f},{radius * math.sin(ang):.3f}")
    return " ".join(pts)


def build_roundabout(out_dir, radius=20.0):
    """Single-lane four-arm roundabout.

    Circulating traffic has priority (edgePriority). The ego enters from the
    south and exits north; the conflict is the south entry merge, which is the
    same single-conflict-point abstraction as the crossing and the zipper merge.

    The full OD catalogue is emitted so demand tables can be swept, but two
    routes are structurally off-limits as background: SE and SW start on S_in,
    the ego's own insertion lane. Which routes actually receive demand is set by
    ``background_routes`` on the scenario spec, not here.
    """
    os.makedirs(out_dir, exist_ok=True)
    R = float(radius)
    L = ARM_LENGTH
    speed_app = 13.89
    speed_circ = 8.33

    nodes = f"""
<nodes>
    <node id="nS" x="0.0" y="{-R}" type="priority" rightOfWay="edgePriority"/>
    <node id="nE" x="{R}" y="0.0" type="priority" rightOfWay="edgePriority"/>
    <node id="nN" x="0.0" y="{R}" type="priority" rightOfWay="edgePriority"/>
    <node id="nW" x="{-R}" y="0.0" type="priority" rightOfWay="edgePriority"/>
    <node id="S_app" x="0.0" y="{-R - L}" type="priority"/>
    <node id="E_app" x="{R + L}" y="0.0" type="priority"/>
    <node id="N_app" x="0.0" y="{R + L}" type="priority"/>
    <node id="W_app" x="{-R - L}" y="0.0" type="priority"/>
</nodes>
"""

    se = _arc_shape(R, -90.0, 0.0)
    en = _arc_shape(R, 0.0, 90.0)
    nw = _arc_shape(R, 90.0, 180.0)
    ws = _arc_shape(R, 180.0, 270.0)
    edges = f"""
<edges>
    <edge id="circ_SE" from="nS" to="nE" numLanes="1" speed="{speed_circ}" priority="2" shape="{se}"/>
    <edge id="circ_EN" from="nE" to="nN" numLanes="1" speed="{speed_circ}" priority="2" shape="{en}"/>
    <edge id="circ_NW" from="nN" to="nW" numLanes="1" speed="{speed_circ}" priority="2" shape="{nw}"/>
    <edge id="circ_WS" from="nW" to="nS" numLanes="1" speed="{speed_circ}" priority="2" shape="{ws}"/>
    <edge id="S_in"  from="S_app" to="nS"   numLanes="1" speed="{speed_app}" priority="1"/>
    <edge id="S_out" from="nS"    to="S_app" numLanes="1" speed="{speed_app}" priority="1"/>
    <edge id="E_in"  from="E_app" to="nE"   numLanes="1" speed="{speed_app}" priority="1"/>
    <edge id="E_out" from="nE"    to="E_app" numLanes="1" speed="{speed_app}" priority="2"/>
    <edge id="N_in"  from="N_app" to="nN"   numLanes="1" speed="{speed_app}" priority="1"/>
    <edge id="N_out" from="nN"    to="N_app" numLanes="1" speed="{speed_app}" priority="2"/>
    <edge id="W_in"  from="W_app" to="nW"   numLanes="1" speed="{speed_app}" priority="1"/>
    <edge id="W_out" from="nW"    to="W_app" numLanes="1" speed="{speed_app}" priority="2"/>
</edges>
"""

    node_file = os.path.join(out_dir, "roundabout.nod.xml")
    edge_file = os.path.join(out_dir, "roundabout.edg.xml")
    net_file = os.path.join(out_dir, "roundabout.net.xml")
    _write(node_file, nodes)
    _write(edge_file, edges)
    _netconvert(node_file, edge_file, net_file)

    routes = f"""
<routes>
    <vType id="background" accel="2.6" decel="4.5" emergencyDecel="6.0" sigma="0.5"
           length="5.0" minGap="2.5" maxSpeed="{speed_app}" tau="1.0" carFollowModel="IDM"
           actionStepLength="0.8" speedFactor="normc(1.0,0.1,0.8,1.2)"/>
    <vType id="bg_contest" accel="2.6" decel="4.5" emergencyDecel="4.5" sigma="0.5"
           length="5.0" minGap="2.5" maxSpeed="{speed_app}" tau="0.9" carFollowModel="IDM"
           actionStepLength="1.0" speedFactor="normc(1.05,0.1,0.85,1.25)"
           jmIgnoreFoeProb="1.0" jmIgnoreFoeSpeed="30.0" jmTimegapMinor="0"
           jmCrossingGap="0" color="0.2,0.4,1"/>
    <vType id="ego" accel="3.0" decel="6.0" emergencyDecel="9.0" sigma="0.0"
           length="5.0" minGap="2.0" maxSpeed="{speed_app}" tau="1.0" carFollowModel="IDM"
           color="1,0.4,0"/>

    <!-- Ego -->
    <route id="SN" edges="S_in circ_SE circ_EN N_out"/>
    <!-- Through (180°) -->
    <route id="WE" edges="W_in circ_WS circ_SE E_out"/>
    <route id="EW" edges="E_in circ_EN circ_NW W_out"/>
    <route id="NS" edges="N_in circ_NW circ_WS S_out"/>
    <!-- Right turns (90°) -->
    <route id="WS" edges="W_in circ_WS S_out"/>
    <route id="EN" edges="E_in circ_EN N_out"/>
    <route id="NW" edges="N_in circ_NW W_out"/>
    <!-- SE/SW start on S_in, the ego's own insertion lane. Defined for OD
         sweeps but never assigned background demand: traffic here blocks the
         ego before it can negotiate the merge. -->
    <route id="SE" edges="S_in circ_SE E_out"/>
    <!-- Left turns (270°) -->
    <route id="WN" edges="W_in circ_WS circ_SE circ_EN N_out"/>
    <route id="ES" edges="E_in circ_EN circ_NW circ_WS S_out"/>
    <route id="NE" edges="N_in circ_NW circ_WS circ_SE E_out"/>
    <route id="SW" edges="S_in circ_SE circ_EN circ_NW W_out"/>
</routes>
"""
    _write(os.path.join(out_dir, "roundabout.rou.xml"), routes)

    sumocfg = """
<configuration>
    <input>
        <net-file value="roundabout.net.xml"/>
        <route-files value="roundabout.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <step-length value="0.2"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
        <collision.action value="warn"/>
        <collision.check-junctions value="true"/>
        <collision.mingap-factor value="0"/>
        <default.speeddev value="0.1"/>
    </processing>
    <report>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
</configuration>
"""
    _write(os.path.join(out_dir, "roundabout.sumocfg"), sumocfg)
    return net_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["cross", "merge", "roundabout", "all"],
                        default="all")
    parser.add_argument("--out_root", default=SCENARIO_DIR)
    args = parser.parse_args()

    built = []
    if args.scenario in ("cross", "all"):
        built.append(build_unsignalized_cross(os.path.join(args.out_root, "unsignalized_cross")))
    if args.scenario in ("merge", "all"):
        built.append(build_merge(os.path.join(args.out_root, "merge")))
    if args.scenario in ("roundabout", "all"):
        built.append(build_roundabout(os.path.join(args.out_root, "roundabout")))

    for path in built:
        print(f"built {path}")


if __name__ == "__main__":
    sys.exit(main())
