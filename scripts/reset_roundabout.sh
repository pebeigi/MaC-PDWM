#!/usr/bin/env bash
# Full clean rebuild of the roundabout scenario.
#
# The previous roundabout run is not recoverable by retraining: its background
# OD put traffic on S_in (the ego's own insertion lane) and on circ_EN (the
# ego's exit path), which cut reactive opportunities from ~3.8 to ~1.6 per
# episode. Every arm was trained and scored on an environment where the
# influence channel had largely been removed, so the artefacts are discarded
# rather than reused.
#
# The OD now in mac/envs/sumo_planning_env.py was selected by
# scripts/tune_roundabout_od.py against method-independent gates: rule-based
# feasibility, branch-oracle headroom, and reactive-opportunity count.
#
# The kernel arm is now a *fair* baseline: mac.fit_kernel estimates the channel
# coefficients from the same offline dataset the world model sees, instead of
# reading env.drivers. The privileged form is still available as --belief oracle.
#
#   NGPU=4 ./scripts/reset_roundabout.sh                       # everything
#   STAGE=gate ./scripts/reset_roundabout.sh                   # one stage
#   STAGE=from:planners NGPU=4 ./scripts/reset_roundabout.sh   # resume onward
#
# Stages: reset collect dataset worldmodel kernel gate planners analyze
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv-mac/bin/python}
LOGS=${LOGS:-logs}
ARCHIVE=${ARCHIVE:-data/mac/archive_ra_reset_$(date +%Y%m%d_%H%M)}
EPOCHS=${EPOCHS:-120}
ITERS=${ITERS:-60}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
NGPU=${NGPU:-4}
WORKERS=${WORKERS:-16}
PAIRED_EPISODES=${PAIRED_EPISODES:-150}
ONPOLICY_EPISODES=${ONPOLICY_EPISODES:-100}
STAGE=${STAGE:-all}

RAW=data/mac/raw_roundabout
NPZ=data/mac/roundabout.npz
WM=data/mac/world_model_roundabout.pt
WM_HIST=data/mac/world_model_history_roundabout.pt
KERNEL=data/mac/kernel_roundabout.json
OUT=data/mac/planner_roundabout

# Channel used for collection and for the closed-loop planners. Identical
# across arms, so no arm sees a channel the others do not.
CHANNEL=${CHANNEL:-"--beta_intent 2.5 --beta_margin 0.3 --intent_window 0.8 --type_probs 0.1,0.1,0.8"}
EVAL_CHANNEL=${EVAL_CHANNEL:-"--beta_intent 2.5 --beta_margin 0.3"}
PLAN_FLAGS=${PLAN_FLAGS:-"--plan_action --commit_steps 10 --courtesy_grace_steps 3"}
# Shift SNR was ~0.2 on the interventional gate (0.75 m shift vs 11 m spread);
# exclude it so the planner is not fed noise geometry/history cannot produce.
DIFFUSION_BLOCKS=${DIFFUSION_BLOCKS:-"--belief_blocks risk,clear,intent"}

# Strengthened channel supervision. Note --all_trajectory: restricting the
# trajectory loss to interventional rows (~38% of samples) left the diffusion
# model's futures more diffuse than the history-only model's, so its sample
# spread (~12 m) buried the plan-induced shift (~0.5 m) and the shift feature
# was unusable. The intent head stays interventional-only, which is what the
# causal identification actually requires; the trajectory head does not.
WM_EXTRA=${WM_EXTRA:-"--type_weight 6.0 --delta_weight 3.0 --labelled_traj_weight 6.0 --all_trajectory"}
# Control kept for comparison: trajectory loss on interventional rows only.
WM_VARIANT=${WM_VARIANT:-data/mac/world_model_roundabout_intervtraj.pt}

mkdir -p "$LOGS"

launch_planner() {
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES="$gpu" $PY -m mac.train_planner "$@" &
  sleep 0.5
  while [ "$(jobs -r | wc -l)" -ge "$NGPU" ]; do sleep 2; done
}

reset() {
  echo "=== archive + remove every roundabout artefact → $ARCHIVE ==="
  mkdir -p "$ARCHIVE"
  for path in "$NPZ" "$WM" "$WM_HIST" "$KERNEL" \
              data/mac/wm_eval_roundabout.json \
              data/mac/paired_ci_roundabout.json \
              data/mac/branch_oracle_roundabout.json \
              data/mac/plan_info_roundabout.json; do
    [ -e "$path" ] && mv "$path" "$ARCHIVE/" || true
  done
  if [ -d "$OUT" ]; then mv "$OUT" "$ARCHIVE/planner_roundabout"; fi
  if [ -d "$RAW" ]; then mv "$RAW" "$ARCHIVE/raw_roundabout"; fi
  mkdir -p "$RAW" "$OUT"
  echo "archived; raw and planner dirs recreated empty"
}

collect() {
  echo "=== RA: fresh collection on the fixed OD ==="
  mkdir -p "$RAW"
  # Paired branches give the interventional pairs the Delta loss needs.
  for s in $(seq 1 "$WORKERS"); do
    ( $PY -m mac.data.collect --paired --out_dir "$RAW" \
        --episodes "$PAIRED_EPISODES" --scenario roundabout \
        --seed "$((5000 + s))" $CHANNEL
    ) > "$LOGS/ra_reset_collect_paired_$s.log" 2>&1 &
    sleep 0.3
  done
  wait
  # Unpaired episodes broaden the observational history distribution.
  for s in $(seq 1 "$WORKERS"); do
    ( $PY -m mac.data.collect --out_dir "$RAW" \
        --episodes "$ONPOLICY_EPISODES" --scenario roundabout \
        --seed "$((7000 + s))" $CHANNEL
    ) > "$LOGS/ra_reset_collect_obs_$s.log" 2>&1 &
    sleep 0.3
  done
  wait
  echo "raw episode count: $(ls "$RAW" | wc -l)"
}

dataset() {
  echo "=== RA: rebuild dataset ==="
  $PY -m mac.data.build_dataset --raw_dir "$RAW" --out "$NPZ" \
      2>&1 | tee "$LOGS/ra_reset_dataset.log"
}

worldmodel() {
  echo "=== RA: train diffusion + history world models ==="
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=0 $PY -m mac.train_world_model \
      --dataset "$NPZ" --out "$WM" --epochs "$EPOCHS" $WM_EXTRA \
      2>&1 | tee "$LOGS/ra_reset_wm.log" &
  CUDA_VISIBLE_DEVICES=1 $PY -m mac.train_world_model \
      --dataset "$NPZ" --out "$WM_HIST" --epochs "$EPOCHS" --drop_plan \
      --type_weight 6.0 --labelled_traj_weight 6.0 \
      2>&1 | tee "$LOGS/ra_reset_wm_hist.log" &
  CUDA_VISIBLE_DEVICES=2 $PY -m mac.train_world_model \
      --dataset "$NPZ" --out "$WM_VARIANT" --epochs "$EPOCHS" \
      --type_weight 6.0 --delta_weight 3.0 --labelled_traj_weight 6.0 \
      --interventional_trajectory \
      2>&1 | tee "$LOGS/ra_reset_wm_intervtraj.log" &
  wait

  for ckpt in "$WM" "$WM_VARIANT"; do
    CUDA_VISIBLE_DEVICES=0 $PY -m mac.calibrate_guidance \
        --dataset "$NPZ" --checkpoint "$ckpt" --write \
        2>&1 | tee "$LOGS/ra_reset_calibrate_$(basename "$ckpt" .pt).log"
  done
}

kernel() {
  echo "=== RA: fit the fair kernel baseline from the same offline data ==="
  $PY -m mac.fit_kernel --data "$NPZ" --out "$KERNEL" \
      2>&1 | tee "$LOGS/ra_reset_kernel.log"
}

gate() {
  echo "=== RA: world-model gate ==="
  # The belief encoder queries the model with synthetic constant-acceleration
  # probe plans, which are open-loop by construction. The interventional split
  # is therefore the distribution the planner actually sees; scoring on the
  # pooled split mixes in observational rows where the plan is confounded with
  # the scene and the intent head was never trained (--interventional_intent).
  $PY - <<'PY'
import numpy as np
data = np.load("data/mac/roundabout.npz", allow_pickle=True)
mask = data["interventional_val"]
out = {}
for key in data.files:
    arr = data[key]
    out[key] = arr[mask] if key.endswith("_val") and arr.shape[0] == mask.shape[0] else arr
np.savez_compressed("data/mac/roundabout_interventional_val.npz", **out)
print(f"interventional val subset: {int(mask.sum())} of {mask.shape[0]} samples")
PY

  $PY -m mac.eval_world_model --dataset "$NPZ" --checkpoint "$WM" \
      --history_checkpoint "$WM_HIST" --scenario roundabout \
      $EVAL_CHANNEL --json data/mac/wm_eval_roundabout.json \
      2>&1 | tee "$LOGS/ra_reset_eval_wm.log"

  $PY -m mac.eval_world_model --dataset data/mac/roundabout_interventional_val.npz \
      --checkpoint "$WM" --history_checkpoint "$WM_HIST" --scenario roundabout \
      $EVAL_CHANNEL --json data/mac/wm_eval_roundabout_interv.json \
      2>&1 | tee "$LOGS/ra_reset_eval_wm_interv.log"

  PYTHONPATH=. $PY scripts/compare_intent_arms.py --dataset "$NPZ" \
      --checkpoint "$WM" --history_checkpoint "$WM_HIST" --kernel "$KERNEL" \
      --json data/mac/intent_arms_roundabout.json \
      2>&1 | tee "$LOGS/ra_reset_intent_arms.log"

  $PY - <<'PY'
import json
wm = json.load(open("data/mac/wm_eval_roundabout_interv.json"))
arms = json.load(open("data/mac/intent_arms_roundabout.json"))["interventional only"]
ch = wm["channel"]
recovered = ch["tv_model"] / ch["tv_truth"] if ch["tv_truth"] else float("nan")
dif, ker = arms["diffusion p(y|h,u)"], arms["kernel (offline fit)"]
zero = arms["diffusion p(y|h)"]

# The shift feature is the only belief input geometry and history cannot
# produce, but it is averaged over n_samples draws, so it is only usable if it
# clears the standard error of that average.
n_samples = 8
shift = max(abs(v) for v in ch["shifts"])
sigma = ch["spread"]
snr = shift / (sigma / n_samples ** 0.5)

# Best sampling budget the eval reported, so the comparison is not penalised
# by a short denoising chain.
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
if snr == snr:
    print(f"  shift SNR over {n_samples} samples: {snr:.2f} "
          f"(shift {shift:.2f} m vs spread {sigma:.2f} m; "
          f"{'usable' if snr >= 1 else 'NOT resolvable - drop the shift block'})")
if bad:
    print(f"\n  {bad} gate(s) failed.")
PY
}

planners() {
  echo "=== RA: train all planner arms ==="
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
          --scenario roundabout --iterations "$ITERS" --seed "$seed" \
          --out_dir "$OUT" --tag "${belief}_seed${seed}" \
          $extra $CHANNEL $PLAN_FLAGS \
          > "$LOGS/ra_reset_planner_${belief}_${seed}.log" 2>&1
      i=$((i + 1))
    done
  done
  wait
}

analyze() {
  echo "=== RA: paired CI vs geometry ==="
  $PY -m mac.analyze_results --directory "$OUT" \
      --reference geometry --arms diffusion,history,kernel,none \
      --json data/mac/paired_ci_roundabout.json \
      2>&1 | tee "$LOGS/ra_reset_analysis.log"
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
echo "RA_RESET_DONE"
