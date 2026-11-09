"""Harvard--Oxford anatomy priors in subject patch coordinates.

This uses the dataset's existing classical affine registration, never a learned
segmentation teacher or benchmark target. Atlas membership is a population prior,
not a subject segmentation. See campaign/atlas/README.md for provenance.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn


REGION_NAMES = (
    "left_cerebral_white_matter", "left_cerebral_cortex", "left_lateral_ventricle",
    "left_thalamus", "left_caudate", "left_putamen", "left_pallidum", "brain_stem",
    "left_hippocampus", "left_amygdala", "left_accumbens", "right_cerebral_white_matter",
    "right_cerebral_cortex", "right_lateral_ventricle", "right_thalamus", "right_caudate",
    "right_putamen", "right_pallidum", "right_hippocampus", "right_amygdala",
    "right_accumbens",
)
ATLAS_SHAPE = (91, 109, 91)
DEFAULT_ASSET_PATH = "/data/hassan/nanobrain/atlas/harvard_oxford_subcortical.npz"


class AtlasPrior(nn.Module):
    """Frozen 21-region probability atlas, persisted inside model checkpoints.

    The small compressed asset initializes training. A checkpoint can be loaded
    without that asset: the persistent buffers replace the empty constructor
    buffers. Calling an uninitialized atlas raises instead of guessing anatomy.
    """

    region_names = REGION_NAMES
    num_regions = len(REGION_NAMES)

    def __init__(self, asset_path: str | Path | None = None):
        super().__init__()
        path = Path(asset_path or os.environ.get("NANOBRAIN_ATLAS_PATH", DEFAULT_ASSET_PATH))
        self._asset_ready = path.is_file()
        if self._asset_ready:
            with np.load(path, allow_pickle=False) as asset:
                probabilities = np.asarray(asset["probabilities"], dtype=np.uint8)
                inverse_affine = np.asarray(asset["inverse_affine"], dtype=np.float32)
                names = tuple(asset["region_names"].tolist())
            if probabilities.shape != (*ATLAS_SHAPE, self.num_regions) or names != REGION_NAMES:
                raise ValueError("Atlas asset shape/label order does not match Harvard--Oxford.")
            if probabilities.max() > 100:
                raise ValueError("Atlas probabilities must be uint8 percentages in [0,100].")
            probabilities = torch.from_numpy(probabilities.reshape(-1, self.num_regions))
            inverse_affine = torch.from_numpy(inverse_affine)
        else:
            probabilities = torch.zeros((int(np.prod(ATLAS_SHAPE)), self.num_regions), dtype=torch.uint8)
            inverse_affine = torch.eye(4, dtype=torch.float32)
        self.register_buffer("probabilities", probabilities)
        self.register_buffer("inverse_affine", inverse_affine)
        # Centres of eight equal-volume octants; extent=8 places points at +/-2 voxels.
        self.register_buffer("quadrature", torch.tensor(list(itertools.product((-0.25, 0.25), repeat=3)), dtype=torch.float32), persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        supplied = prefix + "probabilities" in state_dict and prefix + "inverse_affine" in state_dict
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if supplied:
            self._asset_ready = True

    @torch.no_grad()
    def patch_weights(
        self,
        coord_centres: Tensor,
        affine: Tensor | np.ndarray,
        mni_affine: Tensor | np.ndarray,
        patch_extent: tuple[float, float, float] | Tensor = (8, 8, 8),
    ) -> Tensor:
        """Return [B,N,21] approximate mean membership in each patch.

        coord_centres: [B,N,3] voxel centres on the *current image grid*.
        affine: [B,4,4] current image voxel -> original RAS-world millimetres.
        mni_affine: [B,4,4] original RAS-world -> MNI152NLin6Asym millimetres.
        patch_extent: [3] or [B,3] patch side lengths in current-grid voxels.

        For ordinary patch indices g with p voxels/side, use centres
        g*p + (p-1)/2, NOT the positional-table index itself. Preserve affine
        crop/pad/reorientation/resampling changes. For Ryan native patches use
        the actual native patch size, including each crop/pad/jitter shift.

        Eight quadrature points approximate patch overlap. At each point the
        2-mm probabilistic atlas is sampled nearest-neighbour. Out-of-atlas and
        non-finite transforms return zero membership. Probabilities are not
        normalized across regions: incomplete/ambiguous coverage stays visible.
        """
        if not self._asset_ready:
            raise RuntimeError("Atlas not initialized: install asset with campaign/atlas/prepare_atlas.py or load a full checkpoint.")
        if mni_affine is None:
            raise ValueError("mni_affine is required for named anatomical regions.")
        if coord_centres.ndim != 3 or coord_centres.shape[-1] != 3:
            raise ValueError("coord_centres must have shape [B,N,3].")
        device = coord_centres.device
        if self.probabilities.device != device:
            raise ValueError("Move AtlasPrior to the coordinate device before querying.")
        with torch.autocast(device_type=device.type, enabled=False):
            centres = coord_centres.float()
            affine = torch.as_tensor(affine, device=device, dtype=torch.float32)
            mni_affine = torch.as_tensor(mni_affine, device=device, dtype=torch.float32)
            if affine.ndim == 2:
                affine = affine.unsqueeze(0)
            if mni_affine.ndim == 2:
                mni_affine = mni_affine.unsqueeze(0)
            transform = self.inverse_affine.float() @ mni_affine @ affine
            transform = transform.expand(centres.shape[0], -1, -1)
            valid_transform = torch.isfinite(transform).all(dim=(-1, -2))
            transform = torch.nan_to_num(transform)
            extent = torch.as_tensor(patch_extent, device=device, dtype=torch.float32)
            if extent.ndim == 1:
                extent = extent.unsqueeze(0)
            weights = centres.new_zeros((*centres.shape[:2], self.num_regions))
            for point in self.quadrature:
                xyz = centres + (extent * point)[:, None, :]
                ijk = torch.bmm(xyz, transform[:, :3, :3].transpose(1, 2)) + transform[:, None, :3, 3]
                ijk = ijk.round().long()
                valid = valid_transform[:, None]
                valid = valid & (ijk >= 0).all(dim=-1)
                valid = valid & (ijk[..., 0] < ATLAS_SHAPE[0]) & (ijk[..., 1] < ATLAS_SHAPE[1]) & (ijk[..., 2] < ATLAS_SHAPE[2])
                i = ijk[..., 0].clamp(0, ATLAS_SHAPE[0] - 1)
                j = ijk[..., 1].clamp(0, ATLAS_SHAPE[1] - 1)
                k = ijk[..., 2].clamp(0, ATLAS_SHAPE[2] - 1)
                flat = (i * ATLAS_SHAPE[1] + j) * ATLAS_SHAPE[2] + k
                weights.add_(self.probabilities[flat].float() * valid[..., None])
            return weights.mul_(1.0 / (100 * len(self.quadrature)))
