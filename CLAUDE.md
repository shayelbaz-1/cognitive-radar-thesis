# Cognitive Radar Beam Selection — Thesis Codebase

## Thesis in One Sentence
Use camera-derived perceptual uncertainty (entropy) to select which radar beams to fire,
so a sparse subset of beams recovers ≥99% of full-scan 3D detection performance.

## Repo Structure

```
cognitive-radar-thesis/
├── beam_eval/                  ← CURRENT MAIN PIPELINE (CRN-based)
│   ├── beam_selector/
│   │   ├── ensemble_lss.py     ← 5-head LSS ensemble; computes entropy maps
│   │   └── base.py
│   ├── config.py               ← BeamEvalConfig dataclass
│   ├── dataset.py              ← BeamFilteredDataset (wraps NuscDatasetRadarDet)
│   ├── evaluate.py             ← 3-step pipeline: entropy → beam select → CRN eval
│   ├── radar_filter.py         ← filter_bev_points_by_beams(), project_bev_to_pv()
│   ├── visualize.py            ← Dash web app for interactive visualization
│   └── plot_results.py         ← generates thesis figures from results/beam_eval/
│
├── crn/                        ← git submodule → youngskkim/CRN
│                                  (3 local changes: 2 cpp compat fixes + gen_radar_bev.py)
├── scripts/
│   └── run_crn_eval.py         ← baseline CRN evaluation (no beam filtering)
│
├── prior_work/                 ← LSS/entropy phase (ablation + reference)
│   ├── simulation/             ← information-theoretic cognitive radar simulation
│   ├── lss_ensemble/           ← train.py + inference.py for LSS 5-head ensemble
│   ├── entropy_val/            ← entropy validation and analysis scripts
│   ├── scripts/
│   │   └── fix_coordinates_and_darkness.py
│   └── lift_splat_shoot/       ← git submodule → nv-tlabs/lift-splat-shoot
│
├── thesis/
│   └── thesis_anchor.tex       ← single LaTeX source of truth for the thesis
│
├── results/                    ← LOCAL ONLY (gitignored)
│   ├── beam_eval/              ← CRN pipeline outputs (21GB)
│   │   ├── beam_eval_cache/    ← entropy maps (6019 .npz) + beam selections (.pkl)
│   │   ├── beam_filtered_r18_budget{0,20,40,60,80,100}pct/   ← entropy strategy
│   │   ├── beam_filtered_r18_budget*_uniform/                ← uniform baseline
│   │   ├── beam_filtered_r18_budget*_random/                 ← random baseline
│   │   └── det/CRN_r18_256x704_128x128_4key/                 ← full-radar baseline
│   └── prior_work/             ← LSS/simulation/entropy experiment outputs
│
└── checkpoints/                ← LOCAL ONLY (gitignored)
    ├── crn/                    ← CRN_r18_256x704_128x128_4key.pth
    │                              CRN_r50_256x704_128x128_4key.pth
    ├── lss_ensemble/           ← best_ensemble.pth + epoch checkpoints
    └── pretrained/             ← model525000.pt (LSS base weights)
```

## Key Results (beam_eval, R18 model)

| Strategy | 20% | 40% | 60% | 80% | 100% |
|---|---|---|---|---|---|
| Entropy (ours) NDS | 0.501 | 0.519 | 0.531 | 0.537 | 0.540 |
| Uniform NDS        | 0.464 | 0.501 | 0.521 | 0.530 | 0.540 |
| Random NDS         | 0.464 | 0.496 | 0.514 | 0.530 | 0.540 |
| Full radar baseline: NDS 0.540, mAP 0.446 |
| No radar baseline:  NDS 0.405, mAP 0.295 |

## Running Things

```bash
# All commands from repo root: /home/shayelbaz/repos/cognitive-radar-thesis/

# Plot thesis figures (reads from results/beam_eval/)
python beam_eval/plot_results.py

# Run beam-filtered evaluation (entropy strategy, 40% budget)
python beam_eval/evaluate.py \
    --ensemble_ckpt checkpoints/lss_ensemble/best_ensemble.pth \
    --model r18 \
    --beam_budget_pct 40 \
    --gpus 4

# Baseline CRN eval (full radar)
python scripts/run_crn_eval.py --model r18 --gpus 4

# Interactive visualization (Dash app)
python beam_eval/visualize.py --data_root data/nuScenes
```

## Data
NuScenes dataset lives at: `/home/shayelbaz/repos/data_nuscenes/`
Pre-processed radar BEV: inside `crn/data/nuScenes/radar_bev_filter/`
The `config.data_root` default is `"data/nuScenes"` (relative to repo root).

## Checkpoints
- `checkpoints/lss_ensemble/best_ensemble_USED_FOR_RESULTS.pth` — **the exact checkpoint
  used to generate all entropy maps and beam_eval results (20/40/60/80%).** Verified via
  meta.json in results/beam_eval/beam_eval_cache/. Run: full_trainval_5heads_from_scratch,
  2026-01-04_15-23-47.
- `checkpoints/lss_ensemble/best_ensemble.pth` — different checkpoint (do not use for
  reproducing results).
- `checkpoints/crn/CRN_r18_256x704_128x128_4key.pth` — CRN R18 model used for all evals.

## Git Submodules
After a fresh clone, initialize submodules with:
```bash
git submodule update --init --recursive
```
- `crn/` — upstream CRN by youngskkim. Has 3 local compat changes (NOT committed upstream).
- `prior_work/lift_splat_shoot/` — upstream LSS by nv-tlabs. Unmodified.

## Path Conventions
- `_REPO_ROOT` in any script = `cognitive-radar-thesis/` (computed via `os.path`)
- `beam_eval/` imports from `beam_eval.*` (not `beam_filtered_eval.*`)
- `ensemble_lss.py` adds `prior_work/simulation/` and `prior_work/lift_splat_shoot/` to sys.path
- CRN imports (`datasets.*`, `models.*`, `exps.*`) resolved from `crn/` on sys.path

## Core Thesis Claims
1. Camera entropy is a valid proxy for radar beam utility
2. Entropy-guided selection outperforms random and uniform at every budget
3. 80% beam budget recovers ≥99% of full-scan NDS

## Ablation Studies (prior_work/)
- Total entropy vs epistemic-only entropy as beam score
- Depth distribution entropy vs output entropy
- MFA attention weights as alternative signal
- Weather/lighting breakdown (day/night/rain via scene_conditions.py)
