"""Azimuth-based BEV radar filtering and BEV-to-PV projection.

The two public functions mirror the CRN preprocessing pipeline
(``gen_radar_bev.py`` / ``gen_radar_pv.py``) but operate at runtime on
beam-filtered points instead of reading pre-computed files.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from pyquaternion import Quaternion

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

MIN_DISTANCE = 0.1
MAX_DISTANCE = 100.0
IMG_SHAPE = (900, 1600)


def filter_bev_points_by_beams(
    points_bev: np.ndarray,
    selected_azimuths: List[float],
    beam_width: float,
    min_range: float = 1.0,
    max_range: float = 100.0,
) -> np.ndarray:
    """Keep only BEV radar points that fall inside selected beam cones.

    BEV points are in the LiDAR sensor frame (X=right, Y=forward).
    Azimuth is measured from the forward (Y) axis via ``atan2(x, y)``.

    Args:
        points_bev: ``(N, 7)`` array — x, y, z, rcs, vx_comp, vy_comp, sweep.
        selected_azimuths: Azimuth angles (degrees) of the beams to keep.
        beam_width: Full cone width of each beam in degrees.
        min_range / max_range: Range gate in metres.

    Returns:
        Filtered ``(M, 7)`` array (M ≤ N).
    """
    if len(points_bev) == 0 or len(selected_azimuths) == 0:
        return points_bev[:0]

    x, y = points_bev[:, 0], points_bev[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.degrees(np.arctan2(x, y))

    range_mask = (r >= min_range) & (r <= max_range)

    half_bw = beam_width / 2
    beam_mask = np.zeros(len(points_bev), dtype=bool)
    for az in selected_azimuths:
        ang_dist = np.abs(theta - az)
        ang_dist = np.minimum(ang_dist, 360 - ang_dist)
        beam_mask |= ang_dist <= half_bw

    return points_bev[range_mask & beam_mask]


def project_bev_to_pv(
    points_bev: np.ndarray,
    lidar_calibrated_sensor: dict,
    lidar_ego_pose: dict,
    cam_calibrated_sensor: dict,
    cam_ego_pose: dict,
    img_shape: Tuple[int, int] = IMG_SHAPE,
) -> np.ndarray:
    """Project BEV radar points into a single camera's perspective view.

    Replicates the exact transform chain from ``gen_radar_pv.py``:

        lidar-sensor → ego → global → cam-ego → cam-sensor → image plane

    Args:
        points_bev: ``(N, 7)`` BEV array (x, y, z, rcs, vx, vy, sweep).
        lidar_calibrated_sensor / lidar_ego_pose: Calibration & ego pose of the
            LiDAR keyframe that produced the BEV points.
        cam_calibrated_sensor / cam_ego_pose: Calibration & ego pose of the
            target camera.
        img_shape: ``(H, W)`` of the camera image.

    Returns:
        ``(M, 7)`` float32 array: ``[u, v, depth, rcs, vx, vy, sweep]``.
    """
    if len(points_bev) == 0:
        return np.zeros((0, 7), dtype=np.float32)

    # --- lidar sensor → ego → global ---------------------------------
    pc = LidarPointCloud(points_bev[:, :4].T)  # (4, N); col-3 is rcs (placeholder)
    features = points_bev[:, 3:]               # (N, 4): rcs, vx, vy, sweep

    pc.rotate(Quaternion(lidar_calibrated_sensor["rotation"]).rotation_matrix)
    pc.translate(np.array(lidar_calibrated_sensor["translation"]))
    pc.rotate(Quaternion(lidar_ego_pose["rotation"]).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose["translation"]))

    # --- global → camera ego → camera sensor → image ------------------
    pc.translate(-np.array(cam_ego_pose["translation"]))
    pc.rotate(Quaternion(cam_ego_pose["rotation"]).rotation_matrix.T)
    pc.translate(-np.array(cam_calibrated_sensor["translation"]))
    pc.rotate(Quaternion(cam_calibrated_sensor["rotation"]).rotation_matrix.T)

    depths = pc.points[2, :]
    feat_with_depth = np.concatenate([depths[:, None], features], axis=1)

    pts_img = view_points(
        pc.points[:3, :],
        np.array(cam_calibrated_sensor["camera_intrinsic"]),
        normalize=True,
    )

    mask = (
        (depths > MIN_DISTANCE)
        & (depths < MAX_DISTANCE)
        & (pts_img[0] > 1)
        & (pts_img[0] < img_shape[1] - 1)
        & (pts_img[1] > 1)
        & (pts_img[1] < img_shape[0] - 1)
    )

    # [u, v, depth, rcs, vx, vy, sweep]
    return np.concatenate(
        [pts_img[:2, mask].T, feat_with_depth[mask]], axis=1
    ).astype(np.float32)
