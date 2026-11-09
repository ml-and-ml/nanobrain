"""Meaningful geometry and checkpoint checks; run with the campaign torch env."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from atlas import ATLAS_SHAPE, AtlasPrior, DEFAULT_ASSET_PATH, REGION_NAMES


class AtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.atlas = AtlasPrior()
        with np.load(DEFAULT_ASSET_PATH) as asset:
            cls.probabilities = asset["probabilities"]
            cls.atlas_affine = np.linalg.inv(asset["inverse_affine"])

    def test_exact_voxel_centres_and_laterality(self):
        ijk = np.array([np.unravel_index(self.probabilities[..., r].argmax(), ATLAS_SHAPE) for r in range(21)])
        world = ijk @ self.atlas_affine[:3, :3].T + self.atlas_affine[:3, 3]
        weights = self.atlas.patch_weights(torch.tensor(world)[None], np.eye(4), np.eye(4), (0, 0, 0))[0]
        np.testing.assert_allclose(weights.numpy(), self.probabilities[tuple(ijk.T)] / 100, atol=1e-7)
        self.assertLess(world[8, 0], 0, "left hippocampus must have negative MNI x")
        self.assertGreater(world[18, 0], 0, "right hippocampus must have positive MNI x")
        self.assertGreater(weights[8, 8], weights[8, 18])
        self.assertGreater(weights[18, 18], weights[18, 8])

    def test_composed_geometry_matches_native_and_flipped_crop(self):
        generator = torch.Generator().manual_seed(4)
        centres = torch.rand(2, 300, 3, generator=generator) * torch.tensor([130., 160., 130.])
        affine = torch.tensor([[1., 0, 0, -70], [0, 1., 0, -100], [0, 0, 1., -50], [0, 0, 0, 1.]])
        expected = self.atlas.patch_weights(centres, affine, torch.eye(4))
        # An arbitrary world-coordinate change must cancel with the MNI transform.
        world_change = torch.tensor([[1.1, .1, 0, 13], [0, .9, .1, -8], [0, 0, 1.2, 4], [0, 0, 0, 1.]])
        actual = self.atlas.patch_weights(centres, world_change @ affine, torch.linalg.inv(world_change))
        torch.testing.assert_close(actual, expected)
        # Flip/crop a grid; carry its voxel transform in the affine, preserve physical patches.
        voxel_change = torch.tensor([[-1., 0, 0, 149], [0, 1., 0, 7], [0, 0, 1., -3], [0, 0, 0, 1.]])
        inverse = torch.linalg.inv(voxel_change)
        shifted_centres = centres @ inverse[:3, :3].T + inverse[:3, 3]
        actual = self.atlas.patch_weights(shifted_centres, affine @ voxel_change, torch.eye(4))
        torch.testing.assert_close(actual, expected)

    def test_outside_missing_and_nonfinite_transform(self):
        outside = torch.ones(1, 3, 3) * 10000
        self.assertEqual(self.atlas.patch_weights(outside, np.eye(4), np.eye(4)).sum(), 0)
        invalid = np.eye(4)
        invalid[0, 0] = np.nan
        self.assertEqual(self.atlas.patch_weights(torch.zeros_like(outside), np.eye(4), invalid).sum(), 0)
        with self.assertRaises(ValueError):
            self.atlas.patch_weights(outside, np.eye(4), None)

    def test_checkpoint_without_external_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            restored = AtlasPrior(Path(tmp) / "not-present.npz")
            with self.assertRaises(RuntimeError):
                restored.patch_weights(torch.zeros(1, 1, 3), np.eye(4), np.eye(4))
            restored.load_state_dict(self.atlas.state_dict())
            points = torch.tensor([[[-25., -20., -15.], [25., -20., -15.]]])
            expected = self.atlas.patch_weights(points, np.eye(4), np.eye(4))
            actual = restored.patch_weights(points, np.eye(4), np.eye(4))
            torch.testing.assert_close(actual, expected)
            self.assertEqual(restored.region_names, REGION_NAMES)


if __name__ == "__main__":
    unittest.main()
