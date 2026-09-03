#!/usr/bin/env bash
# Improve Cross + Roundabout closed-loop results.
#
# Roundabout (model-capped): more paired commits → rebuild → stronger Δ/intent
#   WM → calibrate → all planner arms → paired CI.
# Crossing (env-capped): stronger influence channel so kernel can beat geometry;
#   retrain diffusion with residual belief blocks; matched baselines; paired CI.
#
#   NGPU=4 ./scripts/improve_cross_roundabout.sh
#   SKIP_COLLECT=1 NGPU=4 ./scripts/improve_cross_roundabout.sh   # reuse raw_*
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv-mac/bin/python}
LOGS=${LOGS:-logs}
ARCHIVE=${ARCHIVE:-data/mac/archive_improve_$(date +%Y%m%d)}
EPOCHS=${EPOCHS:-120}
ITERS=${ITERS:-60}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
NGPU=${NGPU:-4}
WORKERS=${WORKERS:-16}
PAIRED_EPISODES=${PAIRED_EPISODES:-120}
SKIP_COLLECT=${SKIP_COLLECT:-0}
PLAN_FLAGS=${PLAN_FLAGS:-"--plan_action --commit_steps 10 --courtesy_grace_steps 3"}

# Baseline channel (WM training / RA planners).
CHANNEL=${CHANNEL:-"--beta_intent 2.5 --beta_margin 0.3 --intent_window 0.8 --type_probs 0.1,0.1,0.8"}
# Stronger crossing channel: tighter margin, louder plan EMA, more typed drivers.
CROSS_CHANNEL=${CROSS_CHANNEL:-"--beta_intent 4.0 --beta_margin 0.15 --intent_window 0.8 --type_probs 0.15,0.15,0.7"}
# eval_world_model only accepts beta_* (not intent_window / type_probs).
EVAL_CHANNEL=${EVAL_CHANNEL:-"--beta_intent 2.5 --beta_margin 0.3"}
EVAL_CROSS_CHANNEL=${EVAL_CROSS_CHANNEL:-"--beta_intent 4.0 --beta_margin 0.15"}
SKIP_RA_WM=${SKIP_RA_WM:-0}
SKIP_CROSS_WM=${SKIP_CROSS_WM:-0}

mkdir -p "$LOGS" "$ARCHIVE/planner_cross" "$ARCHIVE/planner_roundabout"

launch_planner() {
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES="$gpu" $PY -m mac.train_planner "$@" &
  sleep 0.5
  while [ "$(jobs -r | wc -l)" -ge "$NGPU" ]; do sleep 2; done
}

backup() {
  echo "=== archive previous cross/RA artefacts → $ARCHIVE ==="
  cp -a data/mac/roundabout.npz data/mac/cross.npz "$ARCHIVE/" 2>/dev/null || true
  cp -a data/mac/world_model_roundabout.pt data/mac/world_model_history_roundabout.pt \
        data/mac/world_model_cross.pt data/mac/world_model_history_cross.pt \
        "$ARCHIVE/" 2>/dev/null || true
  cp -a data/mac/wm_eval_roundabout.json data/mac/wm_eval_cross.json \
        data/mac/paired_ci_roundabout.json data/mac/paired_ci_cross.json \
        "$ARCHIVE/" 2>/dev/null || true
  cp -a data/mac/planner_roundabout/metrics_*.json "$ARCHIVE/planner_roundabout/" 2>/dev/null || true
  cp -a data/mac/planner_cross/metrics_*.json "$ARCHIVE/planner_cross/" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Roundabout
# ---------------------------------------------------------------------------

ra_collect() {
  echo "=== RA: extra paired collection ==="
  local raw=data/mac/raw_roundabout
  mkdir -p "$raw"
  for s in $(seq 1 "$WORKERS"); do
    (
      $PY -m mac.data.collect --paired --out_dir "$raw" \
          --episodes "$PAIRED_EPISODES" --scenario roundabout \
          --seed "$((2000 + s))" $CHANNEL
    ) > "$LOGS/collect_ra_paired_improve_$s.log" 2>&1 &
    sleep 0.3
  done
  wait
  echo "roundabout raw count: $(ls "$raw" | wc -l)"
}

ra_dataset_wm() {
  echo "=== RA: rebuild dataset + strong Δ WM ==="
  $PY -m mac.data.build_dataset --raw_dir data/mac/raw_roundabout \
      --out data/mac/roundabout.npz 2>&1 | tee "$LOGS/build_dataset_ra_improve.log"

  local extra="--type_weight 4.0 --delta_weight 2.5 --labelled_traj_weight 5.0 --interventional_trajectory --epochs $EPOCHS"
  CUDA_VISIBLE_DEVICES=0 $PY -m mac.train_world_model \
      --dataset data/mac/roundabout.npz --out data/mac/world_model_roundabout.pt \
      $extra 2>&1 | tee "$LOGS/wm_ra_improve.log" &
  CUDA_VISIBLE_DEVICES=1 $PY -m mac.train_world_model \
      --dataset data/mac/roundabout.npz --out data/mac/world_model_history_roundabout.pt \
      --type_weight 4.0 --labelled_traj_weight 5.0 --epochs "$EPOCHS" --drop_plan \
      2>&1 | tee "$LOGS/wm_hist_ra_improve.log" &
  wait

  CUDA_VISIBLE_DEVICES=0 $PY -m mac.calibrate_guidance \
      --dataset data/mac/roundabout.npz --checkpoint data/mac/world_model_roundabout.pt \
      --write 2>&1 | tee "$LOGS/calibrate_ra_improve.log"

  $PY -m mac.eval_world_model --dataset data/mac/roundabout.npz \
      --checkpoint data/mac/world_model_roundabout.pt \
      --history_checkpoint data/mac/world_model_history_roundabout.pt \
      --scenario roundabout $EVAL_CHANNEL --json data/mac/wm_eval_roundabout.json \
      2>&1 | tee "$LOGS/eval_wm_ra_improve.log"
}

ra_planners() {
  echo "=== RA: retrain planners (none/geometry/kernel/diffusion/history) ==="
  local out=data/mac/planner_roundabout i=0 wm
  mkdir -p "$out"
  for belief in none geometry kernel diffusion history; do
    wm=data/mac/world_model_roundabout.pt
    if [ "$belief" = history ]; then wm=data/mac/world_model_history_roundabout.pt; fi
    for seed in $SEEDS; do
      launch_planner $((i % NGPU)) --belief "$belief" --world_model "$wm" \
          --scenario roundabout --iterations "$ITERS" --seed "$seed" \
          --out_dir "$out" --tag "${belief}_seed${seed}" \
          $CHANNEL $PLAN_FLAGS \
          > "$LOGS/ra_${belief}_${seed}_improve.log" 2>&1
      i=$((i + 1))
    done
  done
  wait
}

# ---------------------------------------------------------------------------
# Crossing
# ---------------------------------------------------------------------------

cross_collect() {
  echo "=== Cross: paired collection under stronger channel ==="
  local raw=data/mac/raw_cross
  mkdir -p "$raw"
  for s in $(seq 1 "$WORKERS"); do
    (
      $PY -m mac.data.collect --paired --out_dir "$raw" \
          --episodes "$PAIRED_EPISODES" --scenario cross \
          --seed "$((3000 + s))" $CROSS_CHANNEL
    ) > "$LOGS/collect_cross_paired_improve_$s.log" 2>&1 &
    sleep 0.3
  done
  wait
  echo "cross raw count: $(ls "$raw" | wc -l)"
}

cross_dataset_wm() {
  echo "=== Cross: rebuild dataset + WM (stronger type/Δ) ==="
  $PY -m mac.data.build_dataset --raw_dir data/mac/raw_cross \
      --out data/mac/cross.npz 2>&1 | tee "$LOGS/build_dataset_cross_improve.log"

  local extra="--type_weight 4.0 --delta_weight 2.0 --labelled_traj_weight 5.0 --interventional_trajectory --epochs $EPOCHS"
  CUDA_VISIBLE_DEVICES=0 $PY -m mac.train_world_model \
      --dataset data/mac/cross.npz --out data/mac/world_model_cross.pt \
      $extra 2>&1 | tee "$LOGS/wm_cross_improve.log" &
  CUDA_VISIBLE_DEVICES=1 $PY -m mac.train_world_model \
      --dataset data/mac/cross.npz --out data/mac/world_model_history_cross.pt \
      --type_weight 4.0 --labelled_traj_weight 5.0 --epochs "$EPOCHS" --drop_plan \
      2>&1 | tee "$LOGS/wm_hist_cross_improve.log" &
  wait

  CUDA_VISIBLE_DEVICES=0 $PY -m mac.calibrate_guidance \
      --dataset data/mac/cross.npz --checkpoint data/mac/world_model_cross.pt \
      --write 2>&1 | tee "$LOGS/calibrate_cross_improve.log"

  $PY -m mac.eval_world_model --dataset data/mac/cross.npz \
      --checkpoint data/mac/world_model_cross.pt \
      --history_checkpoint data/mac/world_model_history_cross.pt \
      --scenario cross $EVAL_CROSS_CHANNEL --json data/mac/wm_eval_cross.json \
      2>&1 | tee "$LOGS/eval_wm_cross_improve.log"
}

cross_planners() {
  echo "=== Cross: planners under stronger channel (+ diffusion belief_blocks) ==="
  local out=data/mac/planner_cross i=0 wm
  mkdir -p "$out"
  for belief in none geometry kernel diffusion history; do
    wm=data/mac/world_model_cross.pt
    if [ "$belief" = history ]; then wm=data/mac/world_model_history_cross.pt; fi
    for seed in $SEEDS; do
      local blocks="" kernel_extra=""
      if [ "$belief" = diffusion ]; then
        blocks="--belief_blocks risk,clear,intent"
      fi
      if [ "$belief" = kernel ]; then
        kernel_extra="--kernel_params data/mac/kernel_cross.json"
      fi
      # shellcheck disable=SC2086
      launch_planner $((i % NGPU)) --belief "$belief" --world_model "$wm" \
          --scenario cross --iterations "$ITERS" --seed "$seed" \
          --out_dir "$out" --tag "${belief}_seed${seed}" \
          $blocks $kernel_extra $CROSS_CHANNEL $PLAN_FLAGS \
          > "$LOGS/cross_${belief}_${seed}_improve.log" 2>&1
      i=$((i + 1))
    done
  done
  wait
}

analyze() {
  echo "=== paired CI analysis ==="
  $PY -m mac.analyze_results --directory data/mac/planner_roundabout \
      --reference geometry --arms diffusion,history,kernel \
      --json data/mac/paired_ci_roundabout.json \
      2>&1 | tee "$LOGS/analysis_ra_improve.log"
  $PY -m mac.analyze_results --directory data/mac/planner_cross \
      --reference geometry --arms diffusion,history,kernel \
      --json data/mac/paired_ci_cross.json \
      2>&1 | tee "$LOGS/analysis_cross_improve.log"

  $PY - <<'PY'
import json
for scene in ("roundabout", "cross"):
    path = f"data/mac/paired_ci_{scene}.json"
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        print(scene, "missing", path)
        continue
    print(f"\n=== {scene} paired Δ vs geometry ===")
    for arm, metrics in sorted(d.get("paired", {}).items()):
        ret = metrics.get("return", {})
        print(f"  {arm}: Δreturn={ret.get('mean'):+.3f}  "
              f"ci95=[{ret.get('ci95', [None, None])[0]:+.3f}, "
              f"{ret.get('ci95', [None, None])[1]:+.3f}]")
    try:
        w = json.load(open(f"data/mac/wm_eval_{scene}.json"))
        ch = w.get("channel", {})
        print(f"  WM labelled={w.get('labelled_frac')} "
              f"intent={w.get('intent_acc')} "
              f"tv_model={ch.get('tv_model')} tv_truth={ch.get('tv_truth')}")
    except FileNotFoundError:
        pass
PY
}

# ---------------------------------------------------------------------------

backup
if [ "$SKIP_COLLECT" != "1" ]; then
  ra_collect
  cross_collect
fi
if [ "$SKIP_RA_WM" != "1" ]; then
  ra_dataset_wm
else
  echo "=== RA: skip dataset/WM; re-run eval only ==="
  $PY -m mac.eval_world_model --dataset data/mac/roundabout.npz \
      --checkpoint data/mac/world_model_roundabout.pt \
      --history_checkpoint data/mac/world_model_history_roundabout.pt \
      --scenario roundabout $EVAL_CHANNEL --json data/mac/wm_eval_roundabout.json \
      2>&1 | tee "$LOGS/eval_wm_ra_improve.log"
fi
ra_planners
if [ "$SKIP_CROSS_WM" != "1" ]; then
  cross_dataset_wm
fi
cross_planners
analyze
echo "IMPROVE_DONE"
