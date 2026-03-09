"""Beam-filtered CRN evaluation pipeline.

Three-step pipeline with persistent caching:

1. **Entropy maps** – computed once per sample using the ensemble model,
   saved to ``<cache_dir>/entropy/<sample_token>.npz``.  Budget-independent,
   so they are reused across all budget runs.  Supports resumption: if the
   process is interrupted, already-computed samples are skipped on restart.
2. **Beam selections** – greedy EIG beam selection from cached entropy maps,
   saved to ``<cache_dir>/beams_<budget>pct.pkl``.  Cheap (no GPU needed),
   cached per budget percentage.
3. **CRN evaluation** – runs CRN with beam-filtered radar.  Results go to a
   budget-specific output directory.

Usage::

    conda activate CRN
    cd crn_fusion
    python beam_filtered_eval/evaluate.py \\
        --ensemble_ckpt path/to/ensemble.pth \\
        --model r18 \\
        --beam_budget_pct 80 \\
        --gpus 4

    # Re-run with a different budget — entropy is reused, only beam
    # selection + CRN evaluation are repeated:
    python beam_filtered_eval/evaluate.py \\
        --ensemble_ckpt path/to/ensemble.pth \\
        --beam_budget_pct 60
"""

import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
from functools import partial

import mmcv
import numpy as np
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
from pytorch_lightning.callbacks.model_summary import ModelSummary
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CRN_FUSION_DIR = os.path.dirname(_THIS_DIR)
_CRN_DIR = os.path.join(_CRN_FUSION_DIR, "CRN")
_REPO_ROOT = os.path.dirname(_CRN_FUSION_DIR)

for _p in (_CRN_DIR, _CRN_FUSION_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from beam_filtered_eval.config import BeamEvalConfig  # noqa: E402
from beam_filtered_eval.beam_selector.ensemble_lss import (  # noqa: E402
    EnsembleBeamSelector,
)
from beam_filtered_eval.dataset import BeamFilteredDataset  # noqa: E402
from datasets.nusc_det_dataset import collate_fn  # noqa: E402


# ---------------------------------------------------------------------------
# Step 1 – Entropy maps  (budget-independent, cached per sample)
# ---------------------------------------------------------------------------

def _entropy_worker(rank, world_size, config, uncached_infos, data_root,
                    entropy_dir):
    """Compute entropy maps for a shard of samples on a single GPU."""
    torch.cuda.set_device(rank)

    chunk = len(uncached_infos) // world_size
    start = rank * chunk
    end = start + chunk if rank < world_size - 1 else len(uncached_infos)

    selector = EnsembleBeamSelector(
        config.ensemble_grid_conf,
        config.ensemble_data_aug_conf,
        model_path=config.ensemble_ckpt,
        beam_width=config.beam_width,
        min_range=config.min_range,
        max_range=config.max_range,
    )

    for info in tqdm(
        uncached_infos[start:end],
        desc=f"Entropy [GPU {rank}]",
        position=rank,
        leave=True,
    ):
        batch = selector.prepare_batch(info["cam_infos"], data_root)
        entropy, belief = selector.compute_entropy_map(batch)
        token = info["sample_token"]
        np.savez(
            os.path.join(entropy_dir, f"{token}.npz"),
            entropy=entropy, belief=belief,
        )


def _validate_entropy_cache(cache_dir, config):
    """Check if the cached entropy maps match the current ensemble ckpt."""
    meta_file = os.path.join(cache_dir, "meta.json")
    ckpt_abs = os.path.abspath(config.ensemble_ckpt)

    if os.path.isfile(meta_file):
        with open(meta_file) as f:
            saved = json.load(f)
        if saved.get("ensemble_ckpt") != ckpt_abs:
            print(f"  WARNING: Ensemble checkpoint changed.\n"
                  f"    cached : {saved.get('ensemble_ckpt')}\n"
                  f"    current: {ckpt_abs}\n"
                  f"  Invalidating entropy cache.")
            entropy_dir = os.path.join(cache_dir, "entropy")
            if os.path.isdir(entropy_dir):
                shutil.rmtree(entropy_dir)
            # Remove stale beam selections too (they depend on entropy)
            for f in os.listdir(cache_dir):
                if f.startswith("beams_") and f.endswith(".pkl"):
                    os.remove(os.path.join(cache_dir, f))

    os.makedirs(cache_dir, exist_ok=True)
    with open(meta_file, "w") as f:
        json.dump({"ensemble_ckpt": ckpt_abs}, f)


def compute_entropy_maps(config, val_infos, data_root, num_gpus, cache_dir):
    """Compute (or load) entropy/belief maps for every validation sample.

    Results are saved per-sample as ``.npz`` files under
    ``<cache_dir>/entropy/``.  Already-cached samples are skipped, so this
    naturally supports resumption after interruptions.

    Returns ``{sample_token: (entropy, belief)}`` for all samples.
    """
    _validate_entropy_cache(cache_dir, config)

    entropy_dir = os.path.join(cache_dir, "entropy")
    os.makedirs(entropy_dir, exist_ok=True)

    cached_tokens = {
        f[:-4] for f in os.listdir(entropy_dir) if f.endswith(".npz")
    }
    uncached = [
        info for info in val_infos
        if info["sample_token"] not in cached_tokens
    ]

    if uncached:
        print(f"  Computing entropy for {len(uncached)} samples "
              f"({len(cached_tokens)} already cached) ...")
        if num_gpus <= 1:
            selector = EnsembleBeamSelector(
                config.ensemble_grid_conf,
                config.ensemble_data_aug_conf,
                model_path=config.ensemble_ckpt,
                beam_width=config.beam_width,
                min_range=config.min_range,
                max_range=config.max_range,
            )
            for info in tqdm(uncached, desc="Computing entropy"):
                batch = selector.prepare_batch(info["cam_infos"], data_root)
                entropy, belief = selector.compute_entropy_map(batch)
                token = info["sample_token"]
                np.savez(
                    os.path.join(entropy_dir, f"{token}.npz"),
                    entropy=entropy, belief=belief,
                )
        else:
            mp.spawn(
                _entropy_worker,
                args=(num_gpus, config, uncached, data_root, entropy_dir),
                nprocs=num_gpus,
            )
        print("  Entropy computation done.")
    else:
        print(f"  All {len(cached_tokens)} entropy maps already cached.")

    # Load all maps
    entropy_maps = {}
    for info in tqdm(val_infos, desc="Loading entropy maps"):
        token = info["sample_token"]
        data = np.load(os.path.join(entropy_dir, f"{token}.npz"))
        entropy_maps[token] = (data["entropy"], data["belief"])
    return entropy_maps


# ---------------------------------------------------------------------------
# Step 2 – Beam selection  (cached per budget percentage)
# ---------------------------------------------------------------------------

def select_beams_cached(config, entropy_maps, cache_dir):
    """Select beams from cached entropy maps, with per-budget caching.

    No GPU model is loaded — only the lightweight greedy EIG algorithm runs.

    Returns ``{sample_token: [selected_azimuths]}``.
    """
    beams_file = os.path.join(
        cache_dir, f"beams_{config.beam_budget_pct:.0f}pct.pkl"
    )

    if os.path.isfile(beams_file):
        print(f"  Loading cached beam selections ({config.beam_budget_pct}%) "
              f"from {beams_file}")
        with open(beams_file, "rb") as f:
            return pickle.load(f)

    print(f"  Selecting beams at {config.beam_budget_pct}% budget "
          f"(no GPU needed) ...")

    selector = EnsembleBeamSelector(
        config.ensemble_grid_conf,
        config.ensemble_data_aug_conf,
        beam_width=config.beam_width,
        min_range=config.min_range,
        max_range=config.max_range,
    )
    candidates = config.candidate_azimuths

    beam_selections = {}
    for token, (entropy, belief) in tqdm(
        entropy_maps.items(), desc="Beam selection"
    ):
        selected = selector.select_beams_from_maps(
            entropy, belief, config.beam_budget_pct, candidates
        )
        beam_selections[token] = selected

    with open(beams_file, "wb") as f:
        pickle.dump(beam_selections, f)
    print(f"  Saved beam selections to {beams_file}")
    return beam_selections


def select_beams_uniform(config, val_infos):
    """Evenly-spaced beams across the FOV — same for all samples."""
    candidates = config.candidate_azimuths
    n = max(1, int(len(candidates) * config.beam_budget_pct / 100))
    indices = np.round(np.linspace(0, len(candidates) - 1, n)).astype(int)
    selected = [float(candidates[i]) for i in indices]
    print(f"  Uniform: {n} evenly-spaced beams from {len(candidates)} candidates")
    return {info["sample_token"]: selected for info in val_infos}


def select_beams_random(config, val_infos, seed=42):
    """Random beam selection per sample with fixed seed."""
    candidates = config.candidate_azimuths
    n = max(1, int(len(candidates) * config.beam_budget_pct / 100))
    print(f"  Random (seed={seed}): {n} beams per sample")
    beam_selections = {}
    for i, info in enumerate(val_infos):
        rng = np.random.default_rng(seed + i)
        chosen = rng.choice(candidates, n, replace=False)
        beam_selections[info["sample_token"]] = [float(a) for a in chosen]
    return beam_selections


def select_beams_none(val_infos):
    """No radar — empty beam list for every sample."""
    print("  No radar: all samples get 0 beams")
    return {info["sample_token"]: [] for info in val_infos}


# ---------------------------------------------------------------------------
# Step 3 – CRN evaluation with beam-filtered radar (multi-GPU)
# ---------------------------------------------------------------------------

def _load_crn_model_class(model_name):
    if model_name == "r18":
        from exps.det.CRN_r18_256x704_128x128_4key import (
            CRNLightningModel,
        )
    elif model_name == "r50":
        from exps.det.CRN_r50_256x704_128x128_4key import (
            CRNLightningModel,
        )
    else:
        raise ValueError(f"Unknown CRN model: {model_name}")
    return CRNLightningModel


def _make_beam_filtered_model(
    model_name, beam_selections, beam_width, min_range, max_range, num_workers,
):
    """Dynamically create a CRN model class with beam-filtered dataset."""
    BaseClass = _load_crn_model_class(model_name)

    class _BeamFilteredCRN(BaseClass):
        def val_dataloader(self):
            ds = BeamFilteredDataset(
                ida_aug_conf=self.ida_aug_conf,
                bda_aug_conf=self.bda_aug_conf,
                rda_aug_conf=self.rda_aug_conf,
                img_backbone_conf=self.backbone_img_conf,
                classes=self.class_names,
                data_root=self.data_root,
                info_paths=self.val_info_paths,
                is_train=False,
                img_conf=self.img_conf,
                load_interval=self.load_interval,
                num_sweeps=self.num_sweeps,
                sweep_idxes=self.sweep_idxes,
                key_idxes=self.key_idxes,
                return_image=self.return_image,
                return_depth=self.return_depth,
                return_radar_pv=self.return_radar_pv,
                remove_z_axis=self.remove_z_axis,
                radar_pv_path="radar_pv_filter",
                beam_selections=beam_selections,
                beam_width=beam_width,
                min_range=min_range,
                max_range=max_range,
            )
            return torch.utils.data.DataLoader(
                ds,
                batch_size=self.batch_size_per_device,
                num_workers=num_workers,
                shuffle=False,
                collate_fn=partial(
                    collate_fn,
                    is_return_image=self.return_image,
                    is_return_depth=self.return_depth,
                    is_return_radar_pv=self.return_radar_pv,
                ),
            )

        def test_dataloader(self):
            return self.val_dataloader()

    return _BeamFilteredCRN


def phase2_crn_evaluation(config, beam_selections, num_gpus=1, num_workers=8,
                          strategy_suffix=""):
    """Evaluate CRN with beam-filtered radar points.

    Always runs on a single GPU because:
    - DDP cannot pickle the dynamically-created model subclass.
    - DP triggers device-mismatch errors inside CRN's bbox decoder.
    Multi-GPU acceleration is applied in Step 1 (entropy) which is the
    bottleneck; CRN inference here is comparatively fast.
    """
    dir_name = (f"beam_filtered_{config.crn_model}_"
                f"budget{config.beam_budget_pct:.0f}pct{strategy_suffix}")
    out_dir = os.path.join("outputs", dir_name)

    ModelClass = _make_beam_filtered_model(
        config.crn_model,
        beam_selections,
        config.beam_width,
        config.min_range,
        config.max_range,
        num_workers,
    )
    model = ModelClass(
        gpus=1,
        data_root=config.data_root,
        batch_size_per_device=1,
        default_root_dir=out_dir,
    )

    trainer = pl.Trainer(
        gpus=1,
        precision=16,
        deterministic=False,
        default_root_dir=out_dir,
        callbacks=[ModelSummary(max_depth=3)],
    )
    trainer.test(model, ckpt_path=config.crn_ckpt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Beam-filtered CRN evaluation pipeline.")
    p.add_argument("--model", choices=["r18", "r50"], default="r18")
    p.add_argument("--beam_budget_pct", type=float, default=20.0)
    p.add_argument("--beam_width", type=float, default=3.0)
    p.add_argument("--azimuth_fov", type=float, default=360.0)
    p.add_argument("--min_range", type=float, default=1.0)
    p.add_argument("--max_range", type=float, default=100.0)
    p.add_argument("--beam_strategy",
                    choices=["entropy", "uniform", "random", "none"],
                    default="entropy",
                    help="Beam selection strategy.")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed (for --beam_strategy random).")
    p.add_argument("--ensemble_ckpt", type=str, default=None,
                    help="Path to ensemble LSS checkpoint "
                         "(required for entropy strategy).")
    p.add_argument("--crn_ckpt", type=str, default=None,
                    help="Path to CRN checkpoint (auto-detected if omitted).")
    p.add_argument("--data_root", type=str, default="data/nuScenes")
    p.add_argument("--cache_dir", type=str, default="outputs/beam_eval_cache",
                    help="Directory for cached entropy maps and beam "
                         "selections.")
    p.add_argument("--gpus", type=int, default=0,
                    help="Number of GPUs (0 = all available).")
    p.add_argument("--num_workers", type=int, default=8,
                    help="DataLoader workers for Phase 2.")
    return p.parse_args()


def main():
    args = parse_args()

    strategy = args.beam_strategy
    if strategy == "entropy" and args.ensemble_ckpt is None:
        raise ValueError("--ensemble_ckpt is required for entropy strategy")

    available = torch.cuda.device_count()
    num_gpus = min(args.gpus, available) if args.gpus > 0 else available
    num_workers = args.num_workers

    ckpt_dir = os.path.join(_REPO_ROOT, "checkpoints", "crn")
    crn_ckpt = args.crn_ckpt or os.path.join(
        ckpt_dir, f"CRN_{args.model}_256x704_128x128_4key.pth")

    config = BeamEvalConfig(
        beam_budget_pct=args.beam_budget_pct,
        azimuth_fov=args.azimuth_fov,
        beam_width=args.beam_width,
        min_range=args.min_range,
        max_range=args.max_range,
        crn_model=args.model,
        crn_ckpt=crn_ckpt,
        ensemble_ckpt=args.ensemble_ckpt or "",
        data_root=args.data_root,
    )

    os.chdir(_CRN_DIR)

    data_root_abs = os.path.join(_CRN_DIR, config.data_root)
    val_info_path = os.path.join(data_root_abs, "nuscenes_infos_val.pkl")
    val_infos = mmcv.load(val_info_path)

    cache_dir = os.path.join(_CRN_DIR, args.cache_dir)

    # Strategy suffix for output directory naming
    strategy_suffix = {"entropy": "", "uniform": "_uniform",
                       "random": "_random", "none": ""}[strategy]

    print("=" * 60)
    print("  Beam-Filtered CRN Evaluation")
    print(f"  Model         : CRN-{config.crn_model.upper()}")
    print(f"  Strategy      : {strategy}")
    print(f"  Beam budget   : {config.beam_budget_pct}%")
    print(f"  Beam width    : {config.beam_width} deg")
    print(f"  Candidates    : {config.num_candidates} (FOV/beam_width)")
    print(f"  FOV           : {config.azimuth_fov} deg")
    print(f"  Val samples   : {len(val_infos)}")
    print(f"  GPUs          : {num_gpus}")
    print(f"  Workers       : {num_workers}")
    if strategy == "random":
        print(f"  Seed          : {args.seed}")
    print("=" * 60)

    # ---- Beam selection based on strategy ----
    if strategy == "none":
        print("\nStrategy: no radar")
        beam_selections = select_beams_none(val_infos)
        # Override budget to 0 for output dir naming
        config.beam_budget_pct = 0.0
        strategy_suffix = ""
    elif strategy == "uniform":
        print(f"\nStrategy: uniform ({config.beam_budget_pct}%)")
        beam_selections = select_beams_uniform(config, val_infos)
    elif strategy == "random":
        print(f"\nStrategy: random ({config.beam_budget_pct}%, seed={args.seed})")
        beam_selections = select_beams_random(config, val_infos, seed=args.seed)
    else:
        # entropy strategy — need Steps 1 & 2
        print("\nStep 1: Entropy maps")
        entropy_maps = compute_entropy_maps(
            config, val_infos, data_root_abs, num_gpus, cache_dir)
        print(f"\nStep 2: Beam selection ({config.beam_budget_pct}%)")
        beam_selections = select_beams_cached(config, entropy_maps, cache_dir)

    counts = [len(v) for v in beam_selections.values()]
    if counts:
        print(f"  {np.mean(counts):.1f} beams/sample (mean), "
              f"range [{min(counts)}, {max(counts)}]")

    # Step 3 — CRN evaluation
    dir_name = (f"beam_filtered_{config.crn_model}_"
                f"budget{config.beam_budget_pct:.0f}pct{strategy_suffix}")
    out_dir = os.path.join(_CRN_DIR, "outputs", dir_name)
    print(f"\nStep 3: CRN evaluation → {out_dir}")
    phase2_crn_evaluation(
        config, beam_selections, num_gpus=num_gpus, num_workers=num_workers,
        strategy_suffix=strategy_suffix)


if __name__ == "__main__":
    main()
