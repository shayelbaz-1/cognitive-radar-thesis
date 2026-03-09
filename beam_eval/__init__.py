"""Beam-filtered CRN evaluation pipeline."""

from .config import BeamEvalConfig
from .radar_filter import filter_bev_points_by_beams, project_bev_to_pv

try:
    from .dataset import BeamFilteredDataset
except ImportError:
    # mmdet3d / CRN deps unavailable – dataset class not needed for viz
    pass
