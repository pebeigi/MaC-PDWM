#!/usr/bin/env bash
# Re-run paper ablations under the new crossing operating point.
#
# Fast defaults: higher CPU parallelism (SUMO-bound), fewer seeds, fewer iters.
# Skips runs whose metrics already reach the requested iteration count (or more).
#
#   NPARALLEL=16 SEEDS="0 1 2 3 4" ITERS=40 STAGE=from:nochannel \
#     ./scripts/run_cross_ablations.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv-mac/bin/python}
LOGS=${LOGS:-logs}
ARCHIVE=${ARCHIVE:-data/mac/archive_cross_ablations_$(date +%Y%m%d_%H%M)}
EPOCHS=${EPOCHS:-120}
ITERS=${ITERS:-40}
SEEDS=${SEEDS:-"0 1 2 3 4"}
NGPU=${NGPU:-4}
# SUMO is CPU-bound; allow many more concurrent planners than GPUs.
NPARALLEL=${NPARALLEL:-16}
STAGE=${STAGE:-all}
MIN_ITER=${MIN_ITER:-$((ITERS - 1))}

WM=data/mac/world_model_cross.pt
WM_HIST=data/mac/world_model_history_cross.pt
WM_IND=data/mac/world_model_independent_cross.pt
KERNEL=data/mac/kernel_cross.json
NPZ=data/mac/cross.npz
MAIN=data/mac/planner_cross

CHANNEL=${CHANNEL:-"--beta_intent 4.0 --beta_margin 0.15 --intent_window 0.8 --type_probs 0.15,0.15,0.7"}
NOCHANNEL=${NOCHANNEL:-"--beta_intent 0.0 --beta_margin 0.15 --intent_window 0.8 --type_probs 0.15,0.15,0.7"}
PLAN_FLAGS=${PLAN_FLAGS:-"--plan_action --commit_steps 10 --courtesy_grace_steps 3"}
DIFFUSION_BLOCKS=${DIFFUSION_BLOCKS:-"--belief_blocks risk,clear,intent"}
WM_EXTRA=${WM_EXTRA:-"--type_weight 6.0 --delta_weight 3.0 --labelled_traj_weight 6.0 --all_trajectory"}

HARD_EVAL=${HARD_EVAL:-"--bg_scale 2.0 --beta_intent 4.0 --beta_margin 0.15 --type_probs 0.05,0.25,0.70"}
SHIFT_EVAL=${SHIFT_EVAL:-"--bg_scale 2.5 --beta_intent 4.0 --beta_margin 0.15 --type_probs 0.15,0.35,0.50"}

mkdir -p "$LOGS"

want() {
  local name=$1
  case "$STAGE" in
    all) return 0 ;;
    from:*)
      local order=(archive nochannel sweep independent hardshift cf)
      local start="${STAGE#from:}"
      local seen=0
      for s in "${order[@]}"; do
        [ "$s" = "$start" ] && seen=1
        [ "$seen" = 1 ] && [ "$s" = "$name" ] && return 0
      done
      return 1
      ;;
    *)
      [[ ",$STAGE," == *",$name,"* ]] && return 0
      return 1
      ;;
  esac
}

# True if metrics_<tag>.json exists and last history iteration >= MIN_ITER.
is_done() {
  local out=$1 tag=$2
  local metrics="$out/metrics_${tag}.json"
  [ -f "$metrics" ] || return 1
  $PY - "$metrics" "$MIN_ITER" <<'PY'
import json, sys
path, need = sys.argv[1], int(sys.argv[2])
hist = json.load(open(path)).get("history") or []
it = hist[-1].get("iteration", -1) if hist else -1
sys.exit(0 if it >= need else 1)
PY
}

# Drop incomplete mid-run artefacts so a fresh launch can overwrite cleanly.
clear_partial() {
  local out=$1 tag=$2
  if is_done "$out" "$tag"; then
    return 0
  fi
  rm -f "$out/metrics_${tag}.json" "$out/policy_${tag}.pt"
}

launch_planner() {
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES="$gpu" $PY -m mac.train_planner "$@" &
  sleep 0.3
  while [ "$(jobs -r | wc -l)" -ge "$NPARALLEL" ]; do sleep 2; done
}

archive() {
  echo "=== archive old ablation dirs → $ARCHIVE ==="
  mkdir -p "$ARCHIVE"
  for d in planner_nochannel planner_sweep planner_independent \
           planner_hard planner_shift; do
    if [ -d "data/mac/$d" ]; then
      mv "data/mac/$d" "$ARCHIVE/$d"
      echo "  moved data/mac/$d"
    fi
  done
  for f in wm_counterfactual_cross.json wm_counterfactual_merge.json \
           wm_counterfactual_roundabout.json world_model_independent_cross.pt; do
    if [ -e "data/mac/$f" ]; then
      mv "data/mac/$f" "$ARCHIVE/$f"
      echo "  moved data/mac/$f"
    fi
  done
}

nochannel() {
  echo "=== β₂=0 control → data/mac/planner_nochannel (resume-aware) ==="
  local out=data/mac/planner_nochannel
  mkdir -p "$out"
  # Drop orphaned partials outside the requested seed set.
  for f in "$out"/metrics_*.json; do
    [ -f "$f" ] || continue
    local tag base seed
    tag=$(basename "$f" | sed 's/^metrics_//;s/\.json$//')
    seed=${tag##*_seed}
    local keep=0
    for s in $SEEDS; do [ "$s" = "$seed" ] && keep=1 && break; done
    if [ "$keep" = 0 ]; then
      echo "  drop out-of-scope $tag"
      rm -f "$out/metrics_${tag}.json" "$out/policy_${tag}.pt"
    fi
  done
  local gpu=0 launched=0 skipped=0
  for seed in $SEEDS; do
    for belief in none history diffusion; do
      local tag="${belief}_seed${seed}"
      if is_done "$out" "$tag"; then
        echo "  skip done $tag"
        skipped=$((skipped + 1))
        continue
      fi
      clear_partial "$out" "$tag"
      local wm="$WM"
      local extra=""
      if [ "$belief" = "history" ]; then
        wm="$WM_HIST"
      elif [ "$belief" = "diffusion" ]; then
        extra="$DIFFUSION_BLOCKS"
      fi
      echo "  launch $tag → gpu$gpu"
      launch_planner "$gpu" \
        --belief "$belief" --world_model "$wm" --scenario cross \
        --iterations "$ITERS" --seed "$seed" --out_dir "$out" \
        $NOCHANNEL $PLAN_FLAGS $extra \
        > "$LOGS/ablation_nochannel_${belief}_s${seed}.log" 2>&1
      gpu=$(( (gpu + 1) % NGPU ))
      launched=$((launched + 1))
    done
  done
  wait
  echo "nochannel: launched=$launched skipped=$skipped metrics=$(ls "$out"/metrics_*.json 2>/dev/null | wc -l)"
}

sweep() {
  echo "=== S-sweep → data/mac/planner_sweep (S=1,4,16; S=8 from main) ==="
  local out=data/mac/planner_sweep
  mkdir -p "$out"
  for seed in $SEEDS; do
    local src="$MAIN/metrics_diffusion_seed${seed}.json"
    local dst="$out/metrics_diffusion_S8_seed${seed}.json"
    if [ -f "$src" ]; then
      $PY - "$src" "$dst" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
blob = json.load(open(src))
blob.setdefault("config", {})["n_samples"] = 8
blob["config"]["tag"] = f"diffusion_S8_seed{blob['config'].get('seed', 0)}"
json.dump(blob, open(dst, "w"), indent=2)
print("wrote", dst)
PY
    fi
  done
  local gpu=0 launched=0 skipped=0
  for S in 1 4 16; do
    for seed in $SEEDS; do
      local tag="diffusion_S${S}_seed${seed}"
      if is_done "$out" "$tag"; then
        echo "  skip done $tag"
        skipped=$((skipped + 1))
        continue
      fi
      clear_partial "$out" "$tag"
      echo "  launch $tag → gpu$gpu"
      launch_planner "$gpu" \
        --belief diffusion --world_model "$WM" --scenario cross \
        --iterations "$ITERS" --seed "$seed" --out_dir "$out" \
        --n_samples "$S" --tag "$tag" \
        $CHANNEL $PLAN_FLAGS $DIFFUSION_BLOCKS \
        > "$LOGS/ablation_sweep_S${S}_s${seed}.log" 2>&1
      gpu=$(( (gpu + 1) % NGPU ))
      launched=$((launched + 1))
    done
  done
  wait
  echo "sweep: launched=$launched skipped=$skipped metrics=$(ls "$out"/metrics_*.json 2>/dev/null | wc -l)"
}

independent() {
  echo "=== independent WM + planners ==="
  if [ ! -f "$NPZ" ]; then
    echo "missing $NPZ"; exit 1
  fi
  if [ ! -f "$WM_IND" ]; then
    echo "  training independent WM..."
    CUDA_VISIBLE_DEVICES=0 $PY -m mac.train_world_model \
      --dataset "$NPZ" --out "$WM_IND" --epochs "$EPOCHS" \
      --independent $WM_EXTRA \
      > "$LOGS/ablation_wm_independent.log" 2>&1
  else
    echo "  reuse existing $WM_IND"
  fi
  echo "independent WM → $WM_IND"

  local out=data/mac/planner_independent
  mkdir -p "$out"
  local gpu=0 launched=0 skipped=0
  for seed in $SEEDS; do
    local tag="diffusion_seed${seed}"
    if is_done "$out" "$tag"; then
      echo "  skip done $tag"
      skipped=$((skipped + 1))
      continue
    fi
    clear_partial "$out" "$tag"
    echo "  launch $tag → gpu$gpu"
    launch_planner "$gpu" \
      --belief diffusion --world_model "$WM_IND" --scenario cross \
      --iterations "$ITERS" --seed "$seed" --out_dir "$out" \
      $CHANNEL $PLAN_FLAGS $DIFFUSION_BLOCKS \
      > "$LOGS/ablation_independent_s${seed}.log" 2>&1
    gpu=$(( (gpu + 1) % NGPU ))
    launched=$((launched + 1))
  done
  wait
  echo "independent: launched=$launched skipped=$skipped metrics=$(ls "$out"/metrics_*.json 2>/dev/null | wc -l)"
}

hardshift() {
  echo "=== zero-shot hard + shift eval of $MAIN ==="
  if [ ! -d "$MAIN" ] || [ "$(ls "$MAIN"/policy_*.pt 2>/dev/null | wc -l)" -lt 1 ]; then
    echo "need saved policies in $MAIN"; exit 1
  fi
  mkdir -p data/mac/planner_hard data/mac/planner_shift
  CUDA_VISIBLE_DEVICES=0 $PY -m mac.eval_shift \
    --planner_dir "$MAIN" --out_dir data/mac/planner_hard \
    --episodes 50 $HARD_EVAL \
    > "$LOGS/ablation_hard.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY -m mac.eval_shift \
    --planner_dir "$MAIN" --out_dir data/mac/planner_shift \
    --episodes 50 $SHIFT_EVAL \
    > "$LOGS/ablation_shift.log" 2>&1 &
  wait
  echo "hard: $(ls data/mac/planner_hard/metrics_*.json 2>/dev/null | wc -l)  shift: $(ls data/mac/planner_shift/metrics_*.json 2>/dev/null | wc -l)"
}

cf() {
  echo "=== counterfactual WM eval (cross / merge / roundabout) ==="
  CUDA_VISIBLE_DEVICES=0 $PY -m mac.eval_counterfactual \
    --scenario cross --world_model "$WM" --history_model "$WM_HIST" \
    --beta_intent 4.0 --beta_margin 0.15 --type_probs 0.15,0.15,0.7 \
    --target_states --episodes 40 \
    --json data/mac/wm_counterfactual_cross.json \
    > "$LOGS/ablation_cf_cross.log" 2>&1 &
  if [ -f data/mac/world_model_merge.pt ]; then
    CUDA_VISIBLE_DEVICES=1 $PY -m mac.eval_counterfactual \
      --scenario merge \
      --world_model data/mac/world_model_merge.pt \
      --history_model data/mac/world_model_history_merge.pt \
      --target_states --episodes 40 \
      --json data/mac/wm_counterfactual_merge.json \
      > "$LOGS/ablation_cf_merge.log" 2>&1 &
  else
    echo "skip merge CF (no world_model_merge.pt)"
  fi
  if [ -f data/mac/world_model_roundabout.pt ]; then
    CUDA_VISIBLE_DEVICES=2 $PY -m mac.eval_counterfactual \
      --scenario roundabout \
      --world_model data/mac/world_model_roundabout.pt \
      --history_model data/mac/world_model_history_roundabout.pt \
      --target_states --episodes 40 \
      --json data/mac/wm_counterfactual_roundabout.json \
      > "$LOGS/ablation_cf_roundabout.log" 2>&1 &
  else
    echo "skip roundabout CF (no world_model_roundabout.pt)"
  fi
  wait
  echo "CF JSONs:"
  ls -1 data/mac/wm_counterfactual_*.json 2>/dev/null || true
}

echo "STAGE=$STAGE NGPU=$NGPU NPARALLEL=$NPARALLEL ITERS=$ITERS SEEDS=[$SEEDS] MIN_ITER=$MIN_ITER"
want archive && archive
want nochannel && nochannel
want sweep && sweep
want independent && independent
want hardshift && hardshift
want cf && cf
echo "=== ablations finished ==="
