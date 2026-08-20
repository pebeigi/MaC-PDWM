#!/usr/bin/env bash
# Experimental pipeline for the paper: three scenarios, five belief arms.
#
#   ./scripts/run_experiments.sh all          # everything, from raw collection
#   ./scripts/run_experiments.sh planners     # just retrain the policies
#   SCENES=cross ./scripts/run_experiments.sh planners    # one scenario
#   ./scripts/run_experiments.sh -h
#
# Override defaults with env vars, e.g. SEEDS="0 1 2" NGPU=2 ITERS=40.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-.venv-mac/bin/python}
LOGS=${LOGS:-logs}
SCENES=${SCENES:-"cross merge roundabout"}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
# Planner arms besides history. history is added automatically when listed here.
BELIEFS=${BELIEFS:-"none geometry kernel diffusion history"}
ITERS=${ITERS:-60}
EPOCHS=${EPOCHS:-120}
NGPU=${NGPU:-4}
# Collection is one SUMO process per episode, so it scales with cores rather
# than memory; WORKERS x EPISODES is the per-scenario episode count.
WORKERS=${WORKERS:-16}
EPISODES=${EPISODES:-500}
PAIRED_EPISODES=${PAIRED_EPISODES:-100}
ORACLE_EPISODES=${ORACLE_EPISODES:-10}

# Influence channel the ego's *plan* can actually move. Decision distance is
# scenario-specific (cross/merge 35 m, roundabout 22 m + approach-edge gate).
# Omit --decision_distance so EnvConfig picks the ScenarioSpec default.
CHANNEL=${CHANNEL:-"--beta_intent 2.5 --beta_margin 0.3 --intent_window 0.8 --type_probs 0.1,0.1,0.8"}
BETAS=${BETAS:-"--beta_intent 2.5 --beta_margin 0.3"}
# The receding commit must be short enough that the policy can still wait out
# a hard contester: 8 s (commit_steps=20) collapsed Cross/RA to a single
# assert-or-nothing action. Merge's win used 10 steps (4 s).
PLAN_FLAGS=${PLAN_FLAGS:-"--plan_action --commit_steps 10 --courtesy_grace_steps 3"}

mkdir -p "$LOGS" paper/figures

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

launch() {
  local gpu=$1; shift
  CUDA_VISIBLE_DEVICES="$gpu" $PY -m mac.train_planner "$@" &
  # Simultaneous traci.start calls race for a free port and one loses with
  # "connection reset by peer"; a short stagger is enough to avoid it.
  sleep 0.5
}

wait_jobs() { wait || true; }

_raw()      { echo "data/mac/raw_$1"; }
_dataset()  { echo "data/mac/$1.npz"; }
_wm()       { echo "data/mac/world_model_$1.pt"; }
_wm_hist()  { echo "data/mac/world_model_history_$1.pt"; }
_planner()  { echo "data/mac/planner_$1"; }

_tp_flag() {
  if [ -n "${TIME_PENALTY:-}" ]; then echo --time_penalty "$TIME_PENALTY"; fi
}

# ---------------------------------------------------------------------------
# per-scenario steps
# ---------------------------------------------------------------------------

_collect() {
  local scen=$1 raw
  raw=$(_raw "$scen")
  rm -rf "$raw"; mkdir -p "$raw"
  for s in $(seq 1 "$WORKERS"); do
    (
      $PY -m mac.data.collect --out_dir "$raw" --episodes "$EPISODES" \
          --scenario "$scen" --seed "$s" $CHANNEL
      $PY -m mac.data.collect --paired --out_dir "$raw" \
          --episodes "$PAIRED_EPISODES" --scenario "$scen" --seed "$s" $CHANNEL
    ) > "$LOGS/collect_${scen}_$s.log" 2>&1 &
    # Simultaneous traci.start calls race for a free port; stagger the launch.
    sleep 0.3
  done
  wait_jobs
  echo "collected $(ls "$raw" | wc -l) episodes into $raw"
}

_dataset_build() {
  local scen=$1
  $PY -m mac.data.build_dataset --raw_dir "$(_raw "$scen")" \
      --out "$(_dataset "$scen")" 2>&1 | tee "$LOGS/build_dataset_${scen}.log"
}

# The gate: only worth training on this data if the plan is informative at all.
_gate() {
  local scen=$1
  $PY -m mac.diagnose_plan_info --dataset "$(_dataset "$scen")" \
      --json "data/mac/plan_info_${scen}.json" 2>&1 | tee "$LOGS/gate_${scen}.log"
  $PY -m mac.eval_branch_oracle --scenario "$scen" \
      --episodes "$ORACLE_EPISODES" $CHANNEL \
      --json "data/mac/branch_oracle_${scen}.json" 2>&1 \
      | tee "$LOGS/branch_oracle_${scen}.log"
}

_worldmodel() {
  local scen=$1 extra=""
  # Roundabout intention labels were ~majority after arc-mixing; extra type
  # weight keeps the head from being drowned by curved-trajectory FDE.
  if [ "$scen" = roundabout ] || [ "$scen" = cross ]; then
    extra="--type_weight 4.0 --delta_weight 1.5 --labelled_traj_weight 5.0"
  fi
  CUDA_VISIBLE_DEVICES=0 $PY -m mac.train_world_model \
      --dataset "$(_dataset "$scen")" --out "$(_wm "$scen")" \
      --epochs "$EPOCHS" $extra 2>&1 | tee "$LOGS/wm_${scen}.log" &
  CUDA_VISIBLE_DEVICES=1 $PY -m mac.train_world_model \
      --dataset "$(_dataset "$scen")" --out "$(_wm_hist "$scen")" \
      --epochs "$EPOCHS" --drop_plan $extra 2>&1 | tee "$LOGS/wm_hist_${scen}.log" &
  wait_jobs
  # Fix the intention head's plan sensitivity on held-out open-loop episodes and
  # store it in the checkpoint, so no planner can choose its own.
  CUDA_VISIBLE_DEVICES=0 $PY -m mac.calibrate_guidance \
      --dataset "$(_dataset "$scen")" --checkpoint "$(_wm "$scen")" \
      --write 2>&1 | tee "$LOGS/calibrate_${scen}.log"
}

_eval_worldmodel() {
  local scen=$1
  $PY -m mac.eval_world_model --dataset "$(_dataset "$scen")" \
      --checkpoint "$(_wm "$scen")" --history_checkpoint "$(_wm_hist "$scen")" \
      --scenario "$scen" $BETAS --json "data/mac/wm_eval_${scen}.json" 2>&1 \
      | tee "$LOGS/eval_wm_${scen}.log"
}

# none / geometry / kernel / diffusion / history, matched evaluation seeds.
# "kernel" is the approximate analytic channel feature. The non-causal exact
# upper bound is evaluated separately by mac.eval_branch_oracle.
_planners() {
  local scen=$1 out i=0 wm
  out=$(_planner "$scen")
  mkdir -p "$out"
  for belief in $BELIEFS; do
    wm=$(_wm "$scen")
    if [ "$belief" = history ]; then wm=$(_wm_hist "$scen"); fi
    for seed in $SEEDS; do
      launch $((i % NGPU)) --belief "$belief" --seed "$seed" --iterations "$ITERS" \
        --world_model "$wm" --out_dir "$out" \
        --scenario "$scen" $CHANNEL $PLAN_FLAGS $(_tp_flag) \
        --tag "${belief}_seed${seed}" \
        > "$LOGS/${scen}_${belief}_$seed.log" 2>&1
      i=$((i + 1))
    done
  done
  wait_jobs
}

# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

stage_collect()          { for s in $SCENES; do _collect "$s"; done; }
stage_dataset()          { for s in $SCENES; do _dataset_build "$s"; done; }
stage_gate()             { for s in $SCENES; do _gate "$s"; done; }
stage_worldmodel()       { for s in $SCENES; do _worldmodel "$s"; done; }
stage_eval_worldmodel()  { for s in $SCENES; do _eval_worldmodel "$s"; done; }
stage_planners()         { for s in $SCENES; do _planners "$s"; done; }
stage_analysis() {
  for s in $SCENES; do
    $PY -m mac.analyze_results --directory "$(_planner "$s")" \
        --json "data/mac/paired_ci_${s}.json" \
        2>&1 | tee "$LOGS/paired_ci_${s}.log"
  done
}
stage_regimes() {
  for s in $SCENES; do
    for bg in 1.0 1.5 2.0; do
      for tp in 0.02 0.06 0.12; do
        $PY -m mac.eval_branch_oracle --scenario "$s" --bg_scale "$bg" \
            --time_penalty "$tp" --episodes "$ORACLE_EPISODES" $CHANNEL \
            --json "data/mac/branch_oracle_${s}_bg${bg}_time${tp}.json" \
            2>&1 | tee "$LOGS/branch_oracle_${s}_bg${bg}_time${tp}.log"
      done
    done
  done
}

stage_all() {
  stage_collect
  stage_dataset
  stage_gate
  stage_worldmodel
  stage_eval_worldmodel
  stage_planners
  stage_analysis
}

stage_figures() {
  mkdir -p paper/figures
  $PY -m mac.plot_setup --out paper/figures 2>&1 | tee "$LOGS/plot_setup.log" || true
  $PY -m mac.plot_curves --logs "$LOGS" --prefix cross \
      --out paper/figures/curves.pdf 2>&1 | tee "$LOGS/plot_curves.log" || true
  $PY -m mac.plot_channel --wm data/mac/wm_eval_cross.json \
      --out paper/figures/channel.pdf 2>&1 | tee -a "$LOGS/plot_curves.log" || true
}

stage_wipe() {
  echo "removing derived results; raw episodes under data/mac/raw_* are kept"
  rm -rf data/mac/planner_* data/mac/*.npz data/mac/*.pt
  rm -f data/mac/wm_eval_*.json data/mac/plan_info_*.json
  rm -rf "$LOGS"; mkdir -p "$LOGS"
  rm -f paper/tab_*.tex paper/results.tex
  echo "wiped. raw data, source, scenarios and the paper tex are untouched."
}

# ---------------------------------------------------------------------------

usage() {
  cat <<EOF
Usage: $0 [stage ...]

Pipeline (each stage runs over \$SCENES = "$SCENES"):
  collect           collect episodes with the mixed behaviour policy
  dataset           build training tensors (tags open-loop episodes)
  gate              is the plan informative about intent in this data?
  worldmodel        train plan-conditioned + history-only WMs, calibrate guidance
  eval_worldmodel   FDE and channel recovery vs the simulator's kernel
  planners          none / geometry / kernel / diffusion / history, \$SEEDS each
  analysis          paired bootstrap confidence intervals across planner seeds
  regimes           traffic-density sweep with the exact branch oracle
  all               all of the above, in order

Paper artefacts:
  figures           setup maps, learning curves, channel plot

Maintenance:
  wipe              delete derived artefacts, keep raw episodes

Environment:
  SCENES="cross merge roundabout"   SEEDS="0..9"   BELIEFS="none geometry kernel diffusion history"
  ITERS=$ITERS
  EPOCHS=$EPOCHS   NGPU=$NGPU   EPISODES=$EPISODES   TIME_PENALTY=(unset)

Examples:
  $0 all
  SCENES=cross SEEDS="0 1 2 3" $0 planners
  NGPU=4 $0 worldmodel eval_worldmodel planners
EOF
}

if [ $# -eq 0 ]; then usage; exit 1; fi

for stage in "$@"; do
  if [ "$stage" = "-h" ] || [ "$stage" = "--help" ]; then usage; exit 0; fi
  echo "=== stage: $stage ==="
  "stage_$stage"
done
echo "=== done ==="
