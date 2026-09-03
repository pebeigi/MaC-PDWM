#!/usr/bin/env bash
# Full rebuild of the crossing scenario pipeline.
#
# The crossing map/OD is structurally fine (SN ego vs EW major traffic). What
# blocked a clear diffusion win was the experiment setup, not the geometry:
#
#   * kernel planners read env.drivers (privileged oracle), not the offline fit
#   * the task was flat (~69-70% success for every arm) under the default channel
#   * WM eval on the pooled split hid strong interventional intent (0.80 vs 0.54)
#
# This script recollects under a stronger influence channel, retrains the world
# model with the roundabout lessons (--all_trajectory, heavy intent/Δ loss),
# fits a fair kernel baseline, gates on the interventional distribution, and
# retrains all planner arms with --kernel_params and diffusion belief_blocks.
#
#   NGPU=4 ./scripts/reset_cross.sh
#   STAGE=from:planners NGPU=4 ./scripts/reset_cross.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv-mac/bin/python}
LOGS=${LOGS:-logs}
ARCHIVE=${ARCHIVE:-data/mac/archive_cross_reset_$(date +%Y%m%d_%H%M)}
EPOCHS=${EPOCHS:-120}
ITERS=${ITERS:-60}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
NGPU=${NGPU:-4}
WORKERS=${WORKERS:-16}
PAIRED_EPISODES=${PAIRED_EPISODES:-150}
ONPOLICY_EPISODES=${ONPOLICY_EPISODES:-100}
STAGE=${STAGE:-all}

RAW=data/mac/raw_cross
NPZ=data/mac/cross.npz
NPZ_INTERV=data/mac/cross_interventional_val.npz
WM=data/mac/world_model_cross.pt
WM_HIST=data/mac/world_model_history_cross.pt
KERNEL=data/mac/kernel_cross.json
OUT=data/mac/planner_cross
SCENARIO=cross

# Stronger channel: louder intent EMA, tighter margin, more reactive drivers.
# Makes negotiation matter more so geometry/history cannot match diffusion.
CHANNEL=${CHANNEL:-"--beta_intent 4.0 --beta_margin 0.15 --intent_window 0.8 --type_probs 0.15,0.15,0.7"}
EVAL_CHANNEL=${EVAL_CHANNEL:-"--beta_intent 4.0 --beta_margin 0.15"}
PLAN_FLAGS=${PLAN_FLAGS:-"--plan_action --commit_steps 10 --courtesy_grace_steps 3"}
DIFFUSION_BLOCKS=${DIFFUSION_BLOCKS:-"--belief_blocks risk,clear,intent"}
WM_EXTRA=${WM_EXTRA:-"--type_weight 6.0 --delta_weight 3.0 --labelled_traj_weight 6.0 --all_trajectory"}

mkdir -p "$LOGS"

launch_planner() {
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES="$gpu" $PY -m mac.train_planner "$@" &
  sleep 0.5
  while [ "$(jobs -r | wc -l)" -ge "$NGPU" ]; do sleep 2; done
}

reset() {
  echo "=== archive + remove cross artefacts → $ARCHIVE ==="
  mkdir -p "$ARCHIVE"
  for path in "$NPZ" "$NPZ_INTERV" "$WM" "$WM_HIST" "$KERNEL" \
              data/mac/wm_eval_cross.json \
              data/mac/wm_eval_cross_interv.json \
              data/mac/intent_arms_cross.json \
              data/mac/paired_ci_cross.json; do
    [ -e "$path" ] && mv "$path" "$ARCHIVE/" || true
  done
  if [ -d "$OUT" ]; then mv "$OUT" "$ARCHIVE/planner_cross"; fi
  if [ -d "$RAW" ]; then mv "$RAW" "$ARCHIVE/raw_cross"; fi
  mkdir -p "$RAW" "$OUT"
  echo "archived; raw and planner dirs recreated empty"
}

collect() {
  echo "=== cross: fresh collection under stronger channel ==="
  mkdir -p "$RAW"
  for s in $(seq 1 "$WORKERS"); do
    ( $PY -m mac.data.collect --paired --out_dir "$RAW" \
        --episodes "$PAIRED_EPISODES" --scenario "$SCENARIO" \
        --seed "$((8000 + s))" $CHANNEL
    ) > "$LOGS/cross_reset_collect_paired_$s.log" 2>&1 &
    sleep 0.3
  done
  wait
  for s in $(seq 1 "$WORKERS"); do
    ( $PY -m mac.data.collect --out_dir "$RAW" \
        --episodes "$ONPOLICY_EPISODES" --scenario "$SCENARIO" \
        --seed "$((9000 + s))" $CHANNEL
    ) > "$LOGS/cross_reset_collect_obs_$s.log" 2>&1 &
    sleep 0.3
  done
  wait
  echo "raw episode count: $(ls "$RAW" | wc -l)"
}

dataset() {
  echo "=== cross: rebuild dataset ==="
  $PY -m mac.data.build_dataset --raw_dir "$RAW" --out "$NPZ" \
      2>&1 | tee "$LOGS/cross_reset_dataset.log"
}

worldmodel() {
  echo "=== cross: train diffusion + history world models ==="
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=0 $PY -m mac.train_world_model \
      --dataset "$NPZ" --out "$WM" --epochs "$EPOCHS" $WM_EXTRA \
      2>&1 | tee "$LOGS/cross_reset_wm.log" &
  CUDA_VISIBLE_DEVICES=1 $PY -m mac.train_world_model \
      --dataset "$NPZ" --out "$WM_HIST" --epochs "$EPOCHS" --drop_plan \
      --type_weight 6.0 --labelled_traj_weight 6.0 \
      2>&1 | tee "$LOGS/cross_reset_wm_hist.log" &
  wait
  CUDA_VISIBLE_DEVICES=0 $PY -m mac.calibrate_guidance \
      --dataset "$NPZ" --checkpoint "$WM" --write \
      2>&1 | tee "$LOGS/cross_reset_calibrate.log"
}

kernel() {
  echo "=== cross: fit fair kernel baseline ==="
  $PY -m mac.fit_kernel --data "$NPZ" --out "$KERNEL" \
      2>&1 | tee "$LOGS/cross_reset_kernel.log"
}

gate() {
  echo "=== cross: world-model gate ==="
  $PY - <<PY
import numpy as np
data = np.load("$NPZ", allow_pickle=True)
mask = data["interventional_val"]
out = {}
for key in data.files:
    arr = data[key]
    out[key] = arr[mask] if key.endswith("_val") and arr.shape[0] == mask.shape[0] else arr
np.savez_compressed("$NPZ_INTERV", **out)
print(f"interventional val subset: {int(mask.sum())} of {mask.shape[0]} samples")
PY

  $PY -m mac.eval_world_model --dataset "$NPZ" --checkpoint "$WM" \
      --history_checkpoint "$WM_HIST" --scenario "$SCENARIO" \
      $EVAL_CHANNEL --json data/mac/wm_eval_cross.json \
      2>&1 | tee "$LOGS/cross_reset_eval_wm.log"

  $PY -m mac.eval_world_model --dataset "$NPZ_INTERV" --checkpoint "$WM" \
      --history_checkpoint "$WM_HIST" --scenario "$SCENARIO" \
      $EVAL_CHANNEL --json data/mac/wm_eval_cross_interv.json \
      2>&1 | tee "$LOGS/cross_reset_eval_wm_interv.log"

  PYTHONPATH=. $PY scripts/compare_intent_arms.py --dataset "$NPZ" \
      --checkpoint "$WM" --history_checkpoint "$WM_HIST" --kernel "$KERNEL" \
      --json data/mac/intent_arms_cross.json \
      2>&1 | tee "$LOGS/cross_reset_intent_arms.log"

  $PY - <<PY
import json
wm = json.load(open("data/mac/wm_eval_cross_interv.json"))
arms = json.load(open("data/mac/intent_arms_cross.json"))["interventional only"]
ch = wm["channel"]
recovered = ch["tv_model"] / ch["tv_truth"] if ch["tv_truth"] else float("nan")
dif, ker = arms["diffusion p(y|h,u)"], arms["kernel (offline fit)"]
zero = arms["diffusion p(y|h)"]
n_samples = 8
shift = max(abs(v) for v in ch["shifts"])
sigma = ch["spread"]
snr = shift / (sigma / n_samples ** 0.5)
diff_fde = min(v["minFDE"] for v in wm["diffusion"].values())
rows = [
    ("channel recovered (interventional)", recovered, 0.85),
    ("intent acc vs majority", dif["acc"] - wm["intent_majority"], 0.02),
    ("intent acc vs fair kernel", dif["acc"] - ker["acc"], 0.0),
    ("intent auc vs fair kernel", dif["auc"] - ker["auc"], 0.0),
    ("plan-conditioning gain (acc)", dif["acc"] - zero["acc"], 0.01),
    ("trajectory: beats history minFDE", wm["history"]["minFDE"] - diff_fde, 0.0),
]
print("\n=== gate (interventional distribution) ===")
bad = 0
for name, value, target in rows:
    ok = value >= target
    bad += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name:42s} {value:+.4f}  (target >= {target})")
print(f"\n  intent acc: diffusion {dif['acc']:.4f} | fair kernel {ker['acc']:.4f} "
      f"| zero-plan {zero['acc']:.4f} | majority {wm['intent_majority']:.4f}")
print(f"  intent auc: diffusion {dif['auc']:.4f} | fair kernel {ker['auc']:.4f}")
print(f"  shift SNR: {snr:.2f} ({'usable' if snr >= 1 else 'drop shift block'})")
if bad:
    print(f"\n  {bad} gate(s) failed — review before trusting planner numbers.")
else:
    print("\n  All gates passed.")
PY
}

planners() {
  echo "=== cross: train all planner arms ==="
  local i=0 wm extra
  mkdir -p "$OUT"
  for belief in none geometry kernel diffusion history; do
    wm="$WM"
    extra=""
    if [ "$belief" = history ]; then wm="$WM_HIST"; fi
    if [ "$belief" = kernel ]; then extra="--kernel_params $KERNEL"; fi
    if [ "$belief" = diffusion ]; then extra="$DIFFUSION_BLOCKS"; fi
    for seed in $SEEDS; do
      # shellcheck disable=SC2086
      launch_planner $((i % NGPU)) --belief "$belief" --world_model "$wm" \
          --scenario "$SCENARIO" --iterations "$ITERS" --seed "$seed" \
          --out_dir "$OUT" --tag "${belief}_seed${seed}" \
          $extra $CHANNEL $PLAN_FLAGS \
          > "$LOGS/cross_reset_planner_${belief}_${seed}.log" 2>&1
      i=$((i + 1))
    done
  done
  wait
}

analyze() {
  echo "=== cross: paired CI vs geometry ==="
  $PY -m mac.analyze_results --directory "$OUT" \
      --reference geometry --arms diffusion,history,kernel,none \
      --json data/mac/paired_ci_cross.json \
      2>&1 | tee "$LOGS/cross_reset_analysis.log"
  $PY - <<'PY'
import json, glob, numpy as np
from collections import defaultdict
rows = defaultdict(list)
for path in glob.glob("data/mac/planner_cross/metrics_*.json"):
    d = json.load(open(path))
    rows[d["config"]["belief"]].append(d["history"][-1])
print("\n=== cross final ranking ===")
ranked = sorted(rows, key=lambda a: -np.mean([x["return"] for x in rows[a]]))
for i, arm in enumerate(ranked, 1):
    r = rows[arm]
    print(f"  {i}. {arm:10s} return={np.mean([x['return'] for x in r]):+.2f}  "
          f"succ={np.mean([x['success_rate'] for x in r]):.3f}")
try:
    ci = json.load(open("data/mac/paired_ci_cross.json"))
    d = ci["paired"]["diffusion"]
    print(f"\n  diffusion vs geometry: Δreturn={d['return']['mean']:+.3f} "
          f"[{d['return']['ci95'][0]:+.3f},{d['return']['ci95'][1]:+.3f}]")
except FileNotFoundError:
    pass
PY
}

STAGES="reset collect dataset worldmodel kernel gate planners analyze"
started=0
for s in $STAGES; do
  case "$STAGE" in
    all)   $s ;;
    from:*) [ "$started" = 1 ] || [ "${STAGE#from:}" = "$s" ] && { started=1; $s; } ;;
    "$s")  $s ;;
  esac
done
echo "CROSS_RESET_DONE"
