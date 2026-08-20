# How to run the full MaC experiment suite

All stages live in `scripts/run_experiments.sh`. From the repo root:

```bash
./scripts/run_experiments.sh --help
```

## Fresh full run (empty `data/mac`)

```bash
rm -rf data/mac logs/v3
NGPU=4 ./scripts/run_experiments.sh all_fresh
```

That collects cross + no-channel + merge, trains all world models, runs every planner arm (main, history, nc, hard, beta, geometry, independent, commit, negotiate), then rebuilds paper tables/figures.

## What is already done (cross, v4)

| Artefact | Path |
|---|---|
| Raw episodes | `data/mac/raw_v3/cross` (8000) |
| Dataset | `data/mac/cross_v4.npz` |
| World model | `data/mac/world_model_v4.pt` |
| WM eval JSON | `data/mac/wm_eval.json` |
| Main planners (10 seeds × 3 arms) | `data/mac/planner_main/` |
| S-sweep | `data/mac/planner_sweep/` |
| No-channel raw (β₂=0) | `data/mac/raw_nc/cross` (8000) |
| Paper tables / figures | `paper/tab_*.tex`, `paper/figures/` |

## Fresh full run status (Aug 2026)

`all_fresh` completed the main, history, no-channel, merge, hard, beta, geometry,
independent, commit, and negotiate arms. Paper tables/figures regenerated from
those artefacts. Remaining soft spots: interventional CF FDE still empty (NaNs);
NC raw collect stopped at ~6773/8000; zero-shot shift eval may still be finishing.

Refresh paper numbers anytime:

```bash
./scripts/run_experiments.sh report figures
# or:
cd paper && tectonic -X compile main.tex
```

## What the results currently support

| Question | Verdict |
|---|---|
| Q1 prediction | Yes — diffusion ≪ CV on minFDE; history ≈ diffusion |
| Q2 counterfactual | Weak — intention monotone but TV≪GT; traj shift ≈ sample spread; CF FDE broken |
| Q3 negotiation | **No** — history ≈ diffusion; geometry captures most lift; plan-as-action / commit do not open a gap |

Honest paper framing: risk-aware WM planning, not demonstrated negotiation.

## Ablations that isolate influence vs prediction vs geometry

1. **History under $\beta_2=0$** (matched no-channel WM): $p(y\mid h)$ vs $p(y\mid h,u)$ with the type-channel off.
2. **$\beta_2$ sweep**: same WMs, env $\beta_2\in\{0,0.9\}$, includes history.
3. **Geometry-only belief**: ego-plan vs CV neighbours, no learned WM.
4. **Independent per-neighbour WM** vs joint diffusion.
5. **Receding-horizon commit** (`commit_steps=5`) so executed $u_{t:t+H}$ matches the queried plan.
6. **Negotiate / plan-as-action**: PPO action = probe, commit $F$ steps.
7. **Q1/Q2** (FDE + interventional `do(u_{1:F})`). Default probes are on-support `{-4,0,+2}`.

```bash
./scripts/run_experiments.sh eval_worldmodel      # Q1
./scripts/run_experiments.sh counterfactual       # Q2 interventional
./scripts/run_experiments.sh report
```

### 1. Matched no-channel (β₂ = 0 data → WM → planners)

```bash
# After the running beta sweep frees GPUs:
./scripts/run_experiments.sh worldmodel_history_nc nochannel_history
# or full matched 2x2 (retrains none/mean/diffusion too):
./scripts/run_experiments.sh dataset_nc worldmodel_nc worldmodel_history_nc nochannel_matched
```

Or step by step:

```bash
.venv-mac/bin/python -m mac.data.build_dataset \
  --raw_dir data/mac/raw_nc/cross --out data/mac/cross_nc.npz

.venv-mac/bin/python -m mac.train_world_model \
  --dataset data/mac/cross_nc.npz --out data/mac/world_model_nc.pt --epochs 120

# then planners (writes data/mac/planner_nochannel/)
SEEDS="0 1 2 3 4 5 6 7 8 9" NGPU=4 \
  ./scripts/run_experiments.sh nochannel_matched
```

### 2. History-only world model (drop plan)

```bash
./scripts/run_experiments.sh worldmodel_history history
```

Equivalent:

```bash
.venv-mac/bin/python -m mac.train_world_model \
  --dataset data/mac/cross_v4.npz --out data/mac/world_model_history.pt \
  --epochs 120 --drop_plan

SEEDS="0 1 2 3 4 5 6 7 8 9" NGPU=4 \
  ./scripts/run_experiments.sh history
```

### 3. Merge scenario (second environment)

```bash
./scripts/run_experiments.sh collect_merge dataset_merge worldmodel_merge main_merge
```

### 4. Rule-based baselines + plan-conditioning isolation

```bash
./scripts/run_experiments.sh baselines belief_sens
# denser traffic, more reactive drivers, stronger channel (5 seeds × 4 arms)
HARD_SEEDS="0 1 2 3 4" NGPU=4 ./scripts/run_experiments.sh hard
# channel-strength sweep (beta_2 = 0 and 0.9)
HARD_SEEDS="0 1 2 3 4" NGPU=4 ./scripts/run_experiments.sh beta
```

### 4b. Geometry / independent WM / commit (after beta GPUs free)

```bash
HARD_SEEDS="0 1 2 3 4" NGPU=4 ./scripts/run_experiments.sh geometry
./scripts/run_experiments.sh worldmodel_independent
HARD_SEEDS="0 1 2 3 4" NGPU=4 ./scripts/run_experiments.sh independent
HARD_SEEDS="0 1 2 3 4" NGPU=4 COMMIT_STEPS=5 ./scripts/run_experiments.sh commit
./scripts/run_experiments.sh worldmodel_history_nc nochannel_history
./scripts/run_experiments.sh eval_worldmodel counterfactual report
```

### 4c. Q3 fix: plan-as-action (run this)

PPO chooses among `{yield, hold, assert}` and holds that accel for \(F=10\) steps. Same action space for none / geometry / history / diffusion. Reactive types now use a 2 s EMA of ego accel, so the executed plan is what moves \(\rho\).

```bash
# fix Q2 table (no GPU needed for long; uses one GPU)
./scripts/run_experiments.sh counterfactual report

# Q3: plan = action (5 seeds × 4 arms)
HARD_SEEDS="0 1 2 3 4" NGPU=4 ITERS=60 \
  ./scripts/run_experiments.sh negotiate
./scripts/run_experiments.sh report
```

### 5. Rebuild paper artefacts

```bash
./scripts/run_experiments.sh report figures
```

## One-shot for everything still missing

```bash
# After raw_nc is ready (it is):
SEEDS="0 1 2 3 4 5 6 7 8 9" NGPU=4 \
  ./scripts/run_experiments.sh all_remaining
```

That runs: `dataset_nc → worldmodel_nc → nochannel_matched → worldmodel_history → history → collect_merge → … → report → figures`.

## Useful knobs

```bash
SEEDS="0 1 2"          # fewer seeds for a smoke run
ITERS=40               # shorter PPO
NGPU=4                 # round-robin CUDA devices
EPOCHS=120             # world-model epochs
PY=.venv-mac/bin/python
```

## Manual module commands (if you prefer)

```bash
# collect
.venv-mac/bin/python -m mac.data.collect --out_dir data/mac/raw_v3/cross \
  --episodes 500 --scenario cross --seed 1

# dataset / WM / eval
.venv-mac/bin/python -m mac.data.build_dataset --raw_dir ... --out ...
.venv-mac/bin/python -m mac.train_world_model --dataset ... --out ... --epochs 120
.venv-mac/bin/python -m mac.eval_world_model --dataset ... --checkpoint ... --json data/mac/wm_eval.json

# planner arms
.venv-mac/bin/python -m mac.train_planner --belief none|mean|diffusion|history \
  --world_model data/mac/world_model_v4.pt --seed 0 --iterations 60 \
  --out_dir data/mac/planner_main

# sever channel in the env only (transfer control)
.venv-mac/bin/python -m mac.train_planner --belief diffusion --beta_intent 0.0 ...

# paper
.venv-mac/bin/python -m mac.report --wm data/mac/wm_eval.json --out paper
.venv-mac/bin/python -m mac.plot_curves --logs logs/v3 --prefix main --out paper/figures/curves.pdf
.venv-mac/bin/python -m mac.plot_channel --wm data/mac/wm_eval.json --out paper/figures/channel.pdf
```

## Expected wall-clock (rough, 4 GPUs)

| Stage | Time |
|---|---|
| dataset build | ~5–15 min |
| world model (120 ep) | ~1–2 h |
| 10 seeds × 3 arms planners | ~4–8 h |
| merge collect 8000 ep | ~3–4 h |
| report + figures | < 1 min |
