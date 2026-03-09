"""Abstract BeamSelector interface."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BeamSelector(Protocol):
    """Select radar beam azimuths for a scene based on camera images.

    Any object implementing ``select_beams`` satisfies this interface --
    no inheritance required.
    """

    def select_beams(
        self,
        images: np.ndarray,
        beam_budget_pct: float,
        candidate_azimuths: np.ndarray,
    ) -> List[float]:
        """Return the subset of ``candidate_azimuths`` (in degrees) to scan.

        Args:
            images: Camera images for the scene, shape (N_cams, H, W, 3).
            beam_budget_pct: Percentage of candidate azimuths to select (0-100).
            candidate_azimuths: Array of candidate azimuth angles in degrees,
                uniformly spaced across the sensor FOV.

        Returns:
            List of selected azimuth angles (degrees).
        """
        ...
