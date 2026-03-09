# Entropy Validation Scripts

## Quick Start

### Test a Model

```bash
cd /home/shayelbaz/repos/entropy_validation

# Edit test_entropy_validation.py line 22-23 to set your model paths
# Then run:
python test_entropy_validation.py
```

### Compare Two Models

```bash
python compare_models.py \
    entropy_validation_results_MODEL1 \
    entropy_validation_results_MODEL2
```

## Scripts

### `test_entropy_validation.py`

Comprehensive 5-test validation suite:
1. **Darkness Sensitivity** - Does entropy increase in low-light?
2. **Distance Uncertainty** - Does entropy increase with distance?
3. **Occlusion Response** - Does entropy increase when cameras blocked?
4. **Complexity Correlation** - Does entropy correlate with scene complexity?
5. **Ensemble Disagreement** - Does entropy capture model variance?

**Output**: 
- `entropy_validation_results_MODEL_NAME/` folder with:
  - Test plots (5 PNG files)
  - Statistics (5 JSON files)
  - Example images (`examples/` folder)
  - Summary report

### `compare_models.py`

Side-by-side comparison of two validated models.

**Output**: Comparison table + winner declaration

### `entropy_graph.py`

Visualization utilities for entropy analysis.

### `entropy_of_depth.py`

Tests depth-based entropy (internal LSS uncertainty).

## Configuration

Edit `test_entropy_validation.py` lines 22-24:

```python
BASELINE_PATH = '../checkpoints/model525000.pt'
ENSEMBLE_PATH = '../checkpoints/runs/mini/YOUR_MODEL/checkpoints/model_best.pth'
DATAROOT = '../data/'
```

## Results

Results are automatically saved to:
```
entropy_validation_results_{MODEL_NAME}_{TIMESTAMP}/
├── darkness_comparison.png
├── distance_boxplot.png
├── occlusion_heatmap.png
├── complexity_scatter.png
├── ensemble_disagreement.png
├── summary_report.txt
└── examples/
    ├── lowlight_scene_1.png (comprehensive visualization)
    ├── lowlight_scene_2.png
    ├── ...
    ├── welllit_scene_1.png
    └── ...
```

## Documentation

- `docs/ENTROPY_VALIDATION_ANALYSIS.md` - Detailed analysis of validation tests
- `docs/COORDINATE_FIX_EXPLANATION.md` - BEV coordinate system explanation

