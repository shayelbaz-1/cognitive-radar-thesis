"""5-head LSS ensemble beam selector (conforms to BeamSelector protocol)."""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pyquaternion import Quaternion

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_SIMULATION_DIR = os.path.join(_REPO_ROOT, "simulation")
_LSS_DIR = os.path.join(_REPO_ROOT, "lift_splat_shoot")
for _p in (_SIMULATION_DIR, _LSS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.models import LiftSplatShoot  # noqa: E402
from information_theory import compute_entropy  # noqa: E402

# ---------------------------------------------------------------------------
# Lightweight copy of EnsembleLSS (avoids heavy radar_simulation imports).
# Matches radar_simulation/radar_simulation.py lines 181-194.
# ---------------------------------------------------------------------------

class _EnsembleLSS(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC=1, num_models=5):
        super().__init__()
        self.models = nn.ModuleList(
            [LiftSplatShoot(grid_conf, data_aug_conf, outC=outC) for _ in range(num_models)]
        )

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        return torch.stack(
            [m(x, rots, trans, intrins, post_rots, post_trans) for m in self.models],
            dim=1,
        )


# ---------------------------------------------------------------------------
# Inline polar conversion (avoids pulling in grid.py via radar_sensor.py)
# ---------------------------------------------------------------------------

def _cartesian_to_polar(x, y):
    """(x=right, y=forward) → (range, azimuth_deg from forward/Y-axis)."""
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.degrees(np.arctan2(x, y))
    return r, theta


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class EnsembleBeamSelector:
    """Entropy-based greedy beam selection using a 5-head LSS ensemble.

    Satisfies the :class:`BeamSelector` protocol.

    If *model_path* is ``None`` the neural-network model is **not** loaded,
    which makes construction cheap.  In that mode only
    :meth:`select_beams_from_maps` (operating on pre-computed entropy/belief
    arrays) is available; calling :meth:`compute_entropy_map` or
    :meth:`prepare_batch` will raise.
    """

    def __init__(
        self,
        grid_conf: dict,
        data_aug_conf: dict,
        *,
        model_path: str = None,
        beam_width: float = 3.0,
        min_range: float = 1.0,
        max_range: float = 100.0,
        target_utility_scale: float = 5.0,
        revisit_angle_deg: float = 3.0,
        revisit_penalty: float = 0.3,
        num_models: int = 5,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.beam_width = beam_width
        self.min_range = min_range
        self.max_range = max_range
        self._target_utility_scale = target_utility_scale
        self._revisit_angle_deg = revisit_angle_deg
        self._revisit_penalty = revisit_penalty

        if model_path is not None:
            self.model = _EnsembleLSS(grid_conf, data_aug_conf, outC=1,
                                      num_models=num_models)
            self.model.to(self.device)
            ckpt = torch.load(model_path, map_location=self.device)
            state = {k.replace("module.", ""): v
                     for k, v in ckpt["state_dict"].items()}
            self.model.load_state_dict(state)
            self.model.eval()
        else:
            self.model = None

        # Pre-compute BEV polar grids (used for every beam selection call)
        H = int((grid_conf["ybound"][1] - grid_conf["ybound"][0]) / grid_conf["ybound"][2])
        W = int((grid_conf["xbound"][1] - grid_conf["xbound"][0]) / grid_conf["xbound"][2])
        xs = np.linspace(grid_conf["xbound"][0], grid_conf["xbound"][1], W)
        ys = np.linspace(grid_conf["ybound"][0], grid_conf["ybound"][1], H)
        X, Y = np.meshgrid(xs, ys)
        self._R, self._Theta = _cartesian_to_polar(X, Y)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_batch(self, cam_infos_dict: dict, data_root: str):
        """Convert a CRN info dict's ``cam_infos`` to an LSS-compatible batch.

        Returns a tuple ``(imgs, rots, trans, intrins, post_rots, post_trans)``
        each with a leading batch dim of 1.

        Requires a loaded model (``model_path`` provided at construction).
        """
        if self.model is None:
            raise RuntimeError(
                "prepare_batch requires a loaded model. "
                "Pass model_path to the constructor."
            )
        cams = self.data_aug_conf["cams"]
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]

        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

        normalize_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        normalize_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        imgs, rots, trans_list, intrins = [], [], [], []
        post_rots_list, post_trans_list = [], []

        for cam in cams:
            cd = cam_infos_dict[cam]

            img = Image.open(os.path.join(data_root, cd["filename"]))
            img = img.resize(resize_dims)
            img = img.crop(crop)
            img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
            img_t = (img_t - normalize_mean) / normalize_std

            w, x, y, z = cd["calibrated_sensor"]["rotation"]
            rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
            tran = torch.Tensor(cd["calibrated_sensor"]["translation"])
            intrin = torch.Tensor(cd["calibrated_sensor"]["camera_intrinsic"])

            post_rot = torch.eye(3)
            post_tran = torch.zeros(3)
            post_rot[:2, :2] *= resize
            post_tran[:2] = torch.tensor([-crop[0], -crop[1]], dtype=torch.float32)

            imgs.append(img_t)
            rots.append(rot)
            trans_list.append(tran)
            intrins.append(intrin)
            post_rots_list.append(post_rot)
            post_trans_list.append(post_tran)

        return (
            torch.stack(imgs).unsqueeze(0),
            torch.stack(rots).unsqueeze(0),
            torch.stack(trans_list).unsqueeze(0),
            torch.stack(intrins).unsqueeze(0),
            torch.stack(post_rots_list).unsqueeze(0),
            torch.stack(post_trans_list).unsqueeze(0),
        )

    # ------------------------------------------------------------------
    # BeamSelector protocol
    # ------------------------------------------------------------------

    def compute_entropy_map(
        self,
        batch,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the ensemble and return ``(entropy, belief)`` BEV maps.

        ``batch`` is the tuple returned by :meth:`prepare_batch`.
        Both arrays have shape ``(H, W)`` matching the BEV grid.

        Requires a loaded model (``model_path`` provided at construction).
        """
        if self.model is None:
            raise RuntimeError(
                "compute_entropy_map requires a loaded model. "
                "Pass model_path to the constructor."
            )
        imgs, rots, trans_t, intrins, post_rots, post_trans = batch

        with torch.no_grad():
            preds = self.model(
                imgs.to(self.device),
                rots.to(self.device),
                trans_t.to(self.device),
                intrins.to(self.device),
                post_rots.to(self.device),
                post_trans.to(self.device),
            )
            probs = torch.sigmoid(preds)
            belief = np.fliplr(probs.mean(dim=1)[0, 0].cpu().numpy())

        return compute_entropy(belief), belief

    def select_beams_from_maps(
        self,
        entropy: np.ndarray,
        belief: np.ndarray,
        beam_budget_pct: float,
        candidate_azimuths: np.ndarray,
    ) -> List[float]:
        """Select beams using pre-computed entropy / belief maps."""
        num_beams = max(1, int(len(candidate_azimuths) * beam_budget_pct / 100))

        selected = []  # type: List[float]
        for _ in range(num_beams):
            best = self._pick_best_beam(entropy, belief, selected, candidate_azimuths)
            if best is None:
                break
            selected.append(best)
        return selected

    def select_beams(
        self,
        images,
        beam_budget_pct: float,
        candidate_azimuths: np.ndarray,
    ) -> List[float]:
        """Select beam azimuths via entropy-based greedy EIG.

        ``images`` is the tuple returned by :meth:`prepare_batch`.
        """
        entropy, belief = self.compute_entropy_map(images)
        return self.select_beams_from_maps(
            entropy, belief, beam_budget_pct, candidate_azimuths)

    # ------------------------------------------------------------------
    # Greedy EIG (adapted from radar_simulation.py lines 714-799)
    # ------------------------------------------------------------------

    def _pick_best_beam(self, entropy, belief, used, candidates):
        used_set = set(used)
        utility = 1.0 + self._target_utility_scale * belief
        eig_map = entropy * utility

        best_az, best_eig = None, -np.inf
        for az in candidates:
            if az in used_set:
                continue

            penalty = 1.0
            for u in used:
                if abs(az - u) < self._revisit_angle_deg:
                    penalty = self._revisit_penalty
                    break

            ang_dist = np.abs(self._Theta - az)
            ang_dist = np.minimum(ang_dist, 360 - ang_dist)
            in_beam = (
                (ang_dist <= self.beam_width / 2)
                & (self._R >= self.min_range)
                & (self._R <= self.max_range)
            )
            eig = eig_map[in_beam].sum() * penalty
            if eig > best_eig:
                best_eig = eig
                best_az = az

        return best_az
