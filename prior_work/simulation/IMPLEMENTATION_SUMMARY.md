# Scene-Condition Metrics Implementation Summary

## Overview
Added **additive** per-condition statistics to the radar simulation without removing any existing functionality. Scenes are now classified by environmental conditions (DAY/NIGHT/RAINY) and metrics are aggregated separately for each condition group.

## Changes Made

### 1. New Module: `scene_conditions.py`
- **`SceneTags` dataclass**: Tracks `is_night`, `is_rain`, `is_day` booleans
- **`classify_scene(scene_rec, log_rec)`**: Classifies scenes using NuScenes metadata
  - Searches for keywords in `scene['description']` and `log` fields
  - Night keywords: 'night', 'nighttime', 'dark', 'evening', 'dusk'
  - Rain keywords: 'rain', 'rainy', 'drizzle', 'wet', 'precipitation', 'shower'
  - Defaults to DAY if classification is unclear
- **`groups_for_tags(tags)`**: Returns overlapping condition groups:
  - `DAY_SCENES` or `NIGHT_SCENES` (mutually exclusive)
  - `RAINY_SCENES` (if rain detected)
  - `RAINY_DAY_SCENES` (if rain + day)
  - `RAINY_NIGHT_SCENES` (if rain + night)
- **`get_scene_groups_from_dataset(dataset, idx)`**: Convenience wrapper

### 2. Modified: `radar_simulation.py`
- **Helper Functions**:
  - `_create_empty_results_dict()`: Centralized results dict creation
  - `_append_scene_to_condition_buckets()`: Appends scene metrics to per-condition buckets
  
- **Updated Functions**: `run_strategy_cognitive`, `run_strategy`, `run_camera_only_baseline`
  - Initialize `results_by_condition = {}` alongside global `results`
  - After each scene, classify and append metrics to condition buckets
  - Aggregation includes new `by_condition` field with per-group stats + counts
  
- **Updated `_generate_comparison`**:
  - Extracts `by_condition` to top-level in JSON output
  - Structure: `save_dict[strategy]['by_condition'][condition] = {metrics, std, count}`

### 3. Modified: `prove_entropy.py`
- **Updated `collect_global_statistics()`**:
  - Now returns `(global_data, by_condition)` tuple
  - Tracks per-condition pixel-level statistics alongside global arrays
  
- **Updated Plot Functions**: `plot_sparsification_oracle`, `plot_calibration`, `plot_properties`
  - Added `suffix=''` parameter for filename customization
  - Saves files as `proof_1_sparsification_oracle{suffix}.png`
  
- **Updated `run()`**:
  - Generates global proofs (existing behavior unchanged)
  - Generates per-condition proofs with suffixes (e.g., `_DAY_SCENES`, `_RAINY_NIGHT_SCENES`)

## Output Structure

### JSON Results (`results.json`)
```json
{
  "strategy_name": {
    "mean": { /* all existing metrics */ },
    "std": { /* all existing std */ },
    "by_condition": {
      "DAY_SCENES": {
        "f1_score": 0.85,
        "iou": 0.75,
        /* ... all metrics ... */
        "std": { /* per-metric std */ },
        "count": 42  // number of scenes
      },
      "NIGHT_SCENES": { /* ... */ },
      "RAINY_SCENES": { /* ... */ },
      "RAINY_DAY_SCENES": { /* ... */ },
      "RAINY_NIGHT_SCENES": { /* ... */ }
    }
  }
}
```

### Entropy Proofs
- **Global**: `proof_1_*.png`, `proof_2_*.png`, `proof_3_*.png`
- **Per-condition**: 
  - `proof_1_sparsification_oracle_DAY_SCENES.png`
  - `proof_2_calibration_NIGHT_SCENES.png`
  - `proof_3_full_confusion_matrix_RAINY_SCENES.png`
  - etc.

## Validation Results

### Unit Tests (`test_scene_conditions.py`)
- ✅ Scene classification logic (day/night/rain/combinations)
- ✅ Helper functions (`_create_empty_results_dict`, `_append_scene_to_condition_buckets`)
- ✅ Default-day policy for unclear scenes

### Smoke Tests (`smoke_test.py`)
- ✅ Module imports
- ✅ Entropy computation (mathematical invariance)
- ✅ Ray tracing physics (unchanged)
- ✅ Bayesian fusion (unchanged)

## Key Design Decisions

1. **Overlapping Buckets**: By design, a rainy-day scene contributes to `DAY_SCENES`, `RAINY_SCENES`, and `RAINY_DAY_SCENES`. This matches the requirement for "each of the following" conditions.

2. **Additive Only**: No existing statistics were removed. Global metrics remain unchanged. Per-condition stats are a pure addition.

3. **Default-Day Policy**: If scene classification is unclear (no keywords found), defaults to `DAY_SCENES` for safe aggregation.

4. **Count Tracking**: Each condition includes a `count` field showing the number of scenes, critical for interpreting statistics from small buckets.

## Usage

### Run Simulation with Condition Tracking
```bash
python radar_simulation.py \
  --strategy cognitive \
  --model_path path/to/model.pth \
  --num_test_scenes 100 \
  --seed 42
```

Check `results.json` for `by_condition` section.

### Generate Per-Condition Entropy Proofs
```bash
python prove_entropy.py \
  --model_path path/to/model.pth \
  --num_scenes 50
```

Check `entropy_proofs/` for condition-specific plots.

### Validate Implementation
```bash
python test_scene_conditions.py  # Unit tests
python smoke_test.py             # Mathematical invariance
```

## Notes

- If NuScenes `log` records lack weather fields, classification relies on `scene['description']` text (consistent with "tag there" expectation)
- Empty condition buckets (e.g., if no rainy scenes in test set) will not appear in `by_condition` output
- Per-condition proofs are only generated for buckets with data (count > 0)
