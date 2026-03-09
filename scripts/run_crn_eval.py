"""
CRN Evaluation Runner
=====================
Runs the CRN checkpoint on the nuScenes validation set and reports NDS / mAP.

Usage
-----
From the repo root (/home/shayelbaz/repos):

    # Evaluate CRN-R18 (faster, ~37 M params, expected NDS 54.2 / mAP 44.9)
    python crn_fusion/run_crn_eval.py --model r18

    # Evaluate CRN-R50 (larger, ~61 M params, expected NDS 56.2 / mAP 47.3)
    python crn_fusion/run_crn_eval.py --model r50

    # Single-GPU, batch-size 1 (also measures inference time / FPS)
    python crn_fusion/run_crn_eval.py --model r18 --gpus 1 --batch_size 1

Prerequisites
-------------
See the "Setup" section at the bottom of this file (or the README inside
crn_fusion/CRN/) for the one-time environment and data-preparation steps.

Expected results (nuScenes val set, 4 key-frames, no CBGS):
    CRN-R18  NDS 54.2  mAP 44.9  FPS ~29
    CRN-R50  NDS 56.2  mAP 47.3  FPS ~23
"""

import argparse
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Paths (relative to the repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRN_DIR = os.path.join(REPO_ROOT, "crn_fusion", "CRN")
CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints", "crn")
DATA_ROOT = os.path.join(REPO_ROOT, "data_nuscenes")  # symlink -> /mnt_hdd15tb/…

# Map model variant -> (exp file, checkpoint file)
MODEL_CONFIGS = {
    "r18": (
        os.path.join(CRN_DIR, "exps", "det", "CRN_r18_256x704_128x128_4key.py"),
        os.path.join(CKPT_DIR, "CRN_r18_256x704_128x128_4key.pth"),
    ),
    "r50": (
        os.path.join(CRN_DIR, "exps", "det", "CRN_r50_256x704_128x128_4key.py"),
        os.path.join(CKPT_DIR, "CRN_r50_256x704_128x128_4key.pth"),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a CRN checkpoint on the nuScenes validation set."
    )
    parser.add_argument(
        "--model",
        choices=["r18", "r50"],
        default="r18",
        help="Which backbone to evaluate (default: r18).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (default: 1).  "
             "Set to 4 for multi-GPU evaluation matching the paper.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device batch size (default: 1).  "
             "Use 1 to also get inference-time / FPS measurements.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Override checkpoint path (uses the bundled checkpoint by default).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override data root (default: data/nuScenes symlink inside CRN_DIR).",
    )
    return parser.parse_args()


def check_prereqs(exp_file, ckpt_path, data_symlink):
    """Verify that all required files/folders exist before launching."""
    ok = True

    if not os.path.isfile(exp_file):
        print(f"[ERROR] Experiment file not found: {exp_file}")
        ok = False

    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        ok = False

    # The CRN code looks for its data at data/nuScenes relative to CRN_DIR
    crn_data_dir = os.path.join(CRN_DIR, "data", "nuScenes")
    if not os.path.exists(crn_data_dir):
        print(
            f"[ERROR] nuScenes data directory not found at: {crn_data_dir}\n"
            f"        Please create the symlink first:\n"
            f"        mkdir -p {os.path.join(CRN_DIR, 'data')}\n"
            f"        ln -s {DATA_ROOT} {crn_data_dir}"
        )
        ok = False
    else:
        for required in [
            "nuscenes_infos_val.pkl",
            "depth_gt",
            "radar_pv_filter",
        ]:
            path = os.path.join(crn_data_dir, required)
            if not os.path.exists(path):
                print(
                    f"[ERROR] Required pre-processed data not found: {path}\n"
                    f"        Run the data-preparation steps described in the "
                    f"'Setup' section at the bottom of this file."
                )
                ok = False

    return ok


def main():
    args = parse_args()

    exp_file, default_ckpt = MODEL_CONFIGS[args.model]
    ckpt_path = args.ckpt or default_ckpt

    print("=" * 70)
    print(f"  CRN Evaluation")
    print(f"  Model    : CRN-{args.model.upper()}")
    print(f"  Exp file : {exp_file}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  GPUs     : {args.gpus}")
    print(f"  Batch/dev: {args.batch_size}")
    print("=" * 70)

    if not check_prereqs(exp_file, ckpt_path, DATA_ROOT):
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build the command that the CRN codebase expects:
    #   python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b <B> --gpus <N>
    #
    # We execute it with CRN_DIR as the working directory so that all the
    # relative imports (data/nuScenes, ops, etc.) resolve correctly.
    # ------------------------------------------------------------------
    cmd = [
        sys.executable,
        exp_file,
        "--ckpt_path", ckpt_path,
        "-e",                               # evaluation mode
        "-b", str(args.batch_size),
        "--gpus", str(args.gpus),
    ]

    env = os.environ.copy()
    # Make sure the CRN source tree is on the Python path so that local
    # imports (models, datasets, evaluators, …) work without installation.
    crn_pythonpath = CRN_DIR
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{crn_pythonpath}:{existing_pp}" if existing_pp else crn_pythonpath
    )

    print(f"\nRunning (cwd={CRN_DIR}):\n  {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=CRN_DIR, env=env)
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Setup instructions (one-time, run manually in a terminal)
# ---------------------------------------------------------------------------
SETUP_INSTRUCTIONS = """
===========================================================================
ONE-TIME SETUP
===========================================================================

1. Create and activate the conda environment
-----------------------------------------------
cd /home/shayelbaz/repos/crn_fusion/CRN

conda env create --file CRN.yaml
conda activate CRN

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \\
    -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==1.6.0
mim install mmcv==1.6.0
mim install mmsegmentation==0.28.0
mim install mmdet==2.25.2

# Build the bundled mmdetection3d fork
cd mmdetection3d
pip install -v -e .
cd ..

# Build CRN custom ops (requires a GPU)
python setup.py develop


2. Create the nuScenes data symlink inside CRN
-----------------------------------------------
mkdir -p /home/shayelbaz/repos/crn_fusion/CRN/data
ln -s /mnt_hdd15tb/shayelbaz/nuscenes_full \\
      /home/shayelbaz/repos/crn_fusion/CRN/data/nuScenes


3. Generate annotation info files
-----------------------------------------------
cd /home/shayelbaz/repos/crn_fusion/CRN
python scripts/gen_info.py
# Produces: data/nuScenes/nuscenes_infos_{train,val}.pkl


4. Generate ground-truth depth maps (requires LiDAR keyframes)
-----------------------------------------------
python scripts/gen_depth_gt.py
# Produces: data/nuScenes/depth_gt/


5. Generate radar point clouds in perspective view
-----------------------------------------------
# Step 5a – accumulate sweeps and transform to LiDAR coords
python scripts/gen_radar_bev.py
# Produces: data/nuScenes/radar_bev_filter/   (temporary, can be deleted later)

# Step 5b – project to each camera's perspective view
python scripts/gen_radar_pv.py
# Produces: data/nuScenes/radar_pv_filter/


Final folder structure inside CRN/data/nuScenes:
    nuscenes_infos_train.pkl
    nuscenes_infos_val.pkl
    maps/
    samples/
    sweeps/
    v1.0-trainval/
    depth_gt/
    radar_bev_filter/   <- safe to delete once radar_pv_filter is ready
    radar_pv_filter/


6. Run evaluation (from the repo root)
-----------------------------------------------
conda activate CRN
python crn_fusion/run_crn_eval.py --model r18   # CRN-R18
python crn_fusion/run_crn_eval.py --model r50   # CRN-R50

# For FPS measurement (single GPU, batch-size 1):
python crn_fusion/run_crn_eval.py --model r18 --gpus 1 --batch_size 1
===========================================================================
"""

if __name__ == "__main__":
    main()
