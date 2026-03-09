"""Configuration for beam-filtered CRN evaluation."""

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BeamEvalConfig:
    # Beam selection
    beam_budget_pct: float = 20.0
    azimuth_fov: float = 360.0
    beam_width: float = 3.0
    min_range: float = 1.0
    max_range: float = 100.0

    @property
    def num_candidates(self) -> int:
        """Number of non-overlapping beams that tile the FOV."""
        return int(math.floor(self.azimuth_fov / self.beam_width))

    @property
    def candidate_azimuths(self) -> np.ndarray:
        """Beam-center azimuths (degrees), perfectly tiling the FOV."""
        n = self.num_candidates
        half = self.azimuth_fov / 2
        first = -half + self.beam_width / 2
        last = half - self.beam_width / 2
        return np.linspace(first, last, n)

    # CRN model
    crn_model: str = "r18"
    crn_ckpt: str = ""
    data_root: str = "data/nuScenes"

    # Ensemble selector
    ensemble_ckpt: str = ""
    ensemble_grid_conf: dict = field(default_factory=lambda: {
        "xbound": [-50, 50, 0.5],
        "ybound": [-50, 50, 0.5],
        "zbound": [-10, 10, 20],
        "dbound": [4.0, 45.0, 1.0],
    })
    ensemble_data_aug_conf: dict = field(default_factory=lambda: {
        "resize_lim": (0.193, 0.193),
        "final_dim": (128, 352),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "bot_pct_lim": (0.0, 0.0),
        "cams": [
            "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
        ],
        "Ncams": 6,
        "rand_flip": False,
    })
