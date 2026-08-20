#!/usr/bin/env bash
# Retrain plan-conditioned world models and diffusion/history planners on
# cross and roundabout so we can test whether diffusion beats geometry.
# Geometry / kernel / none metrics are left in place for paired CIs.
# Merge is not touched.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv-mac/bin/python}
LOGS=${LOGS:-logs}
ARCHIVE=${ARCHIVE:-data/mac/archive_aug16}
EPOCHS=${EPOCHS:-120}
ITERS=${ITERS:-60}
NGPU=${NGPU:-4}
export PY LOGS EPOCHS ITERS NGPU
mkdir -p "$LOGS" "$ARCHIVE/planner_cross" "$ARCHIVE/planner_roundabout"

backup() {
  if [ -f data/mac/cross.npz ] && [ ! -f "$ARCHIVE/cross.npz" ]; then
    echo "archiving previous cross/roundabout artefacts to $ARCHIVE"
    cp -a data/mac/cross.npz data/mac/roundabout.npz "$ARCHIVE/" || true
    cp -a data/mac/world_model_cross.pt data/mac/world_model_roundabout.pt \
          data/mac/world_model_history_cross.pt \
          data/mac/world_model_history_roundabout.pt "$ARCHIVE/" || true
    cp -a data/mac/paired_ci_cross.json data/mac/paired_ci_roundabout.json \
          data/mac/wm_eval_cross.json data/mac/wm_eval_roundabout.json \
          "$ARCHIVE/" || true
    cp -a data/mac/planner_cross/metrics_diffusion_seed*.json \
          data/mac/planner_cross/metrics_history_seed*.json \
          "$ARCHIVE/planner_cross/" || true
    cp -a data/mac/planner_roundabout/metrics_diffusion_seed*.json \
          data/mac/planner_roundabout/metrics_history_seed*.json \
          "$ARCHIVE/planner_roundabout/" || true
  fi
}

rebuild() {
  echo "=== rebuild datasets ==="
  $PY -m mac.data.build_dataset --raw_dir data/mac/raw_cross \
      --out data/mac/cross.npz 2>&1 | tee "$LOGS/build_dataset_cross.log"
  $PY -m mac.data.build_dataset --raw_dir data/mac/raw_roundabout \
      --out data/mac/roundabout.npz 2>&1 | tee "$LOGS/build_dataset_roundabout.log"
}

train_world_models() {
  echo "=== train world models ==="
  local extra="--type_weight 4.0 --delta_weight 1.5 --labelled_traj_weight 5.0 --epochs $EPOCHS"
  CUDA_VISIBLE_DEVICES=0 $PY -m mac.train_world_model \
      --dataset data/mac/cross.npz --out data/mac/world_model_cross.pt \
      $extra 2>&1 | tee "$LOGS/wm_cross.log" &
  CUDA_VISIBLE_DEVICES=1 $PY -m mac.train_world_model \
      --dataset data/mac/cross.npz --out data/mac/world_model_history_cross.pt \
      $extra --drop_plan 2>&1 | tee "$LOGS/wm_hist_cross.log" &
  CUDA_VISIBLE_DEVICES=2 $PY -m mac.train_world_model \
      --dataset data/mac/roundabout.npz --out data/mac/world_model_roundabout.pt \
      $extra 2>&1 | tee "$LOGS/wm_roundabout.log" &
  CUDA_VISIBLE_DEVICES=3 $PY -m mac.train_world_model \
      --dataset data/mac/roundabout.npz --out data/mac/world_model_history_roundabout.pt \
      $extra --drop_plan 2>&1 | tee "$LOGS/wm_hist_roundabout.log" &
  wait

  echo "=== calibrate guidance ==="
  CUDA_VISIBLE_DEVICES=0 $PY -m mac.calibrate_guidance \
      --dataset data/mac/cross.npz --checkpoint data/mac/world_model_cross.pt \
      --write 2>&1 | tee "$LOGS/calibrate_cross.log" &
  CUDA_VISIBLE_DEVICES=1 $PY -m mac.calibrate_guidance \
      --dataset data/mac/roundabout.npz --checkpoint data/mac/world_model_roundabout.pt \
      --write 2>&1 | tee "$LOGS/calibrate_roundabout.log" &
  wait

  echo "=== eval world models ==="
  CUDA_VISIBLE_DEVICES=0 $PY -m mac.eval_world_model \
      --dataset data/mac/cross.npz --checkpoint data/mac/world_model_cross.pt \
      --history_checkpoint data/mac/world_model_history_cross.pt \
      --scenario cross --beta_intent 2.5 --beta_margin 0.3 \
      --json data/mac/wm_eval_cross.json 2>&1 | tee "$LOGS/eval_wm_cross.log" &
  CUDA_VISIBLE_DEVICES=1 $PY -m mac.eval_world_model \
      --dataset data/mac/roundabout.npz --checkpoint data/mac/world_model_roundabout.pt \
      --history_checkpoint data/mac/world_model_history_roundabout.pt \
      --scenario roundabout --beta_intent 2.5 --beta_margin 0.3 \
      --json data/mac/wm_eval_roundabout.json 2>&1 | tee "$LOGS/eval_wm_roundabout.log" &
  wait
  $PY - <<'PY'
import json
for scene in ("cross", "roundabout"):
    d = json.load(open(f"data/mac/wm_eval_{scene}.json"))
    ch = d["channel"]
    print(f"{scene}: labelled={d.get('labelled_frac')} "
          f"intent={d.get('intent_acc'):.3f} maj={d.get('intent_majority'):.3f} "
          f"tv_model={ch['tv_model']:.3f} tv_truth={ch['tv_truth']:.3f} "
          f"shift={ch['shifts'][-1] if ch.get('shifts') else None} "
          f"spread={ch.get('spread')}")
PY
}

train_planners() {
  echo "=== retrain diffusion + history planners ==="
  SCENES="cross roundabout" BELIEFS="diffusion history" \
    NGPU="$NGPU" ITERS="$ITERS" ./scripts/run_experiments.sh planners
  echo "=== paired CIs vs frozen geometry ==="
  SCENES="cross roundabout" ./scripts/run_experiments.sh analysis
}

backup
rebuild
train_world_models
train_planners
echo "=== done ==="
