#!/usr/bin/env bash
# Fix roundabout (more paired data + stronger intent/delta WM) and cross
# (intent+shift belief residual on top of CV risk/clear).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv-mac/bin/python}
LOGS=${LOGS:-logs}
CHANNEL=${CHANNEL:-"--beta_intent 2.5 --beta_margin 0.3 --intent_window 0.8 --type_probs 0.1,0.1,0.8"}
PLAN_FLAGS=${PLAN_FLAGS:-"--plan_action --commit_steps 10 --courtesy_grace_steps 3"}
WORKERS=${WORKERS:-40}
PAIRED_EPISODES=${PAIRED_EPISODES:-120}
EPOCHS=${EPOCHS:-120}
ITERS=${ITERS:-60}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
NGPU=${NGPU:-4}

mkdir -p "$LOGS"

echo "=== 1) roundabout: extra paired collection ==="
raw=data/mac/raw_roundabout
mkdir -p "$raw"
for s in $(seq 1 "$WORKERS"); do
  (
    $PY -m mac.data.collect --paired --out_dir "$raw" \
        --episodes "$PAIRED_EPISODES" --scenario roundabout --seed "$((1000 + s))" \
        $CHANNEL
  ) > "$LOGS/collect_roundabout_paired_extra_$s.log" 2>&1 &
  sleep 0.3
done
wait
echo "roundabout raw count: $(ls "$raw" | wc -l)"

echo "=== 2) rebuild roundabout dataset ==="
$PY -m mac.data.build_dataset --raw_dir "$raw" --out data/mac/roundabout.npz \
    2>&1 | tee "$LOGS/build_dataset_roundabout_fix.log"

echo "=== 3) retrain roundabout world models (stronger intent + delta) ==="
CUDA_VISIBLE_DEVICES=0 $PY -m mac.train_world_model \
    --dataset data/mac/roundabout.npz --out data/mac/world_model_roundabout.pt \
    --epochs "$EPOCHS" --token_conditioned --interventional_trajectory \
    --type_weight 2.0 --delta_weight 1.0 \
    2>&1 | tee "$LOGS/wm_roundabout_fix.log" &
CUDA_VISIBLE_DEVICES=1 $PY -m mac.train_world_model \
    --dataset data/mac/roundabout.npz --out data/mac/world_model_history_roundabout.pt \
    --epochs "$EPOCHS" --drop_plan --token_conditioned \
    --type_weight 2.0 \
    2>&1 | tee "$LOGS/wm_hist_roundabout_fix.log" &
wait
CUDA_VISIBLE_DEVICES=0 $PY -m mac.calibrate_guidance \
    --dataset data/mac/roundabout.npz --checkpoint data/mac/world_model_roundabout.pt \
    --write 2>&1 | tee "$LOGS/calibrate_roundabout_fix.log"
$PY -m mac.eval_world_model --dataset data/mac/roundabout.npz \
    --checkpoint data/mac/world_model_roundabout.pt \
    --history_checkpoint data/mac/world_model_history_roundabout.pt \
    --scenario roundabout $CHANNEL --json data/mac/wm_eval_roundabout.json \
    2>&1 | tee "$LOGS/eval_wm_roundabout_fix.log"

echo "=== 4) retrain roundabout planners ==="
out=data/mac/planner_roundabout
mkdir -p "$out"
i=0
for belief in none geometry kernel diffusion history; do
  for seed in $SEEDS; do
    wm=data/mac/world_model_roundabout.pt
    if [ "$belief" = history ]; then wm=data/mac/world_model_history_roundabout.pt; fi
    gpu=$((i % NGPU))
    CUDA_VISIBLE_DEVICES=$gpu $PY -m mac.train_planner \
        --belief "$belief" --world_model "$wm" --scenario roundabout \
        --iterations "$ITERS" --seed "$seed" --out_dir "$out" \
        --tag "${belief}_seed${seed}" $CHANNEL $PLAN_FLAGS \
        > "$LOGS/roundabout_${belief}_${seed}_fix.log" 2>&1 &
    i=$((i + 1))
    sleep 0.5
    # keep a bounded number of concurrent jobs
    while [ "$(jobs -r | wc -l)" -ge "$NGPU" ]; do sleep 2; done
  done
done
wait

echo "=== 5) cross: diffusion residual belief (risk,clear,intent,shift) ==="
out=data/mac/planner_cross
mkdir -p "$out"
i=0
for seed in $SEEDS; do
  gpu=$((i % NGPU))
  CUDA_VISIBLE_DEVICES=$gpu $PY -m mac.train_planner \
      --belief diffusion --world_model data/mac/world_model_cross.pt \
      --scenario cross --iterations "$ITERS" --seed "$seed" --out_dir "$out" \
      --tag "diffusion_seed${seed}" \
      --belief_blocks risk,clear,intent,shift \
      $CHANNEL $PLAN_FLAGS \
      > "$LOGS/cross_diffusion_${seed}_fix.log" 2>&1 &
  i=$((i + 1))
  sleep 0.5
  while [ "$(jobs -r | wc -l)" -ge "$NGPU" ]; do sleep 2; done
done
# matched geometry/history baselines under same PLAN_FLAGS (overwrite if needed)
for belief in geometry history; do
  for seed in $SEEDS; do
    wm=data/mac/world_model_cross.pt
    if [ "$belief" = history ]; then wm=data/mac/world_model_history_cross.pt; fi
    gpu=$((i % NGPU))
    CUDA_VISIBLE_DEVICES=$gpu $PY -m mac.train_planner \
        --belief "$belief" --world_model "$wm" --scenario cross \
        --iterations "$ITERS" --seed "$seed" --out_dir "$out" \
        --tag "${belief}_seed${seed}" $CHANNEL $PLAN_FLAGS \
        > "$LOGS/cross_${belief}_${seed}_fix.log" 2>&1 &
    i=$((i + 1))
    sleep 0.5
    while [ "$(jobs -r | wc -l)" -ge "$NGPU" ]; do sleep 2; done
  done
done
wait

echo "=== 6) paired CI analysis ==="
$PY -m mac.analyze_results --directory data/mac/planner_cross \
    --reference geometry --arms diffusion,history,kernel \
    --json data/mac/paired_ci_cross.json 2>&1 | tee "$LOGS/analysis_cross_fix.log"
$PY -m mac.analyze_results --directory data/mac/planner_roundabout \
    --reference geometry --arms diffusion,history,kernel \
    --json data/mac/paired_ci_roundabout.json 2>&1 | tee "$LOGS/analysis_roundabout_fix.log"

echo "FIX_DONE"
