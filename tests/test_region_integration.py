"""Small CPU integration checks for Ryan native geometry and gradient paths."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from nanobrain.model import ViTMAE3D, fit_slices


class RecordingAtlas(nn.Module):
    num_regions = 21
    region_names = tuple(f"region_{i}" for i in range(21))

    def __init__(self):
        super().__init__()
        self.calls = []

    def patch_weights(self, centres, affine, mni_affine, patch_extent):
        self.calls.append((centres.clone(), affine.clone(), mni_affine.clone(), patch_extent))
        # Regions span the box, with independent overlapping supports.
        weights = centres.new_zeros(*centres.shape[:2], self.num_regions)
        weights[..., 0] = 1.0
        weights[..., 1] = (centres[..., 0] < centres[..., 0].max() / 2).float()
        weights[..., 8] = (centres[..., 1] < centres[..., 1].max() / 2).float()
        return weights


def make_model(mode):
    model = ViTMAE3D(
        native=True, grid_size=(32, 32, 32), patch_size=8,
        depth=1, embed_dim=24, num_heads=4,
        decoder_depth=1, decoder_embed_dim=16, decoder_num_heads=4,
        class_tokens=0, level_weight=0.5, level_mirror=(1.0, 1.5, 2.0),
        min_mask_frac=0.05, region_mode=mode,
    )
    model.atlas_prior = RecordingAtlas()
    # Baseline zero-initialized decoder intentionally blocks encoder gradients
    # on step zero; test later-step paths with an initialized prediction head.
    nn.init.normal_(model.decoder_head.weight, std=0.01)
    return model


@pytest.mark.parametrize("mode", ["regions", "consistency", "coverage"])
def test_native_mixed_spacing_forward_backward_geometry(mode):
    torch.manual_seed(7338)
    model = make_model(mode)
    images = [torch.randn(30, 34, 32), torch.randn(16, 14, 18), torch.randn(32, 32, 32)]
    masks = [torch.ones_like(image, dtype=torch.bool) for image in images]
    spacings = [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (1.0, 1.0, 1.0)]
    affines = [np.diag([*spacing, 1.0]) for spacing in spacings]
    for affine in affines:
        affine[:3, 3] = [-16, -16, -16]
    mni = [np.eye(4) for _ in images]
    loss, state = model(
        images, masks, spacing=spacings, num_visible=16, num_predict=16, num_samples=2,
        affine=affines, mni_affine=mni,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.blocks[0].attn.q.weight.grad).all()
    assert model.blocks[0].attn.q.weight.grad.abs().sum() > 0
    assert (model.region_context is not None) == (mode == "regions")
    if mode == "regions":
        assert model.region_context.region_identity.grad.abs().sum() > 0
        assert model.region_context.feedback_scale.grad.abs() > 0
    if mode == "consistency":
        assert model.region_stats["consistency_loss"] > 0
    for part in state["parts"]:
        order = part["order"]
        assert order.shape == (len(part["ids"]), 2, 32)
        assert all(row.unique().numel() == 32 for row in order.flatten(0, 1))
    # Group order is 1mm -> ids[0,2], then 2mm -> ids[1].
    for call, ids in zip(model.atlas_prior.calls, ([0, 2], [1])):
        centres, fitted, recorded_mni, patch = call
        for row, index in enumerate(ids):
            box = tuple(round(32 / sp) for sp in spacings[index])
            src, dst = fit_slices(images[index].shape, box)
            shift = np.eye(4)
            shift[:3, 3] = [a.start - b.start for a, b in zip(src, dst)]
            np.testing.assert_allclose(fitted[row].numpy(), affines[index] @ shift)
        np.testing.assert_allclose(centres[0, 0].numpy(), (np.asarray(patch) - 1) / 2)
        np.testing.assert_array_equal(recorded_mni[0].numpy(), np.eye(4))


def test_requires_real_registration_and_two_views():
    model = make_model("consistency")
    image = torch.randn(32, 32, 32)
    with pytest.raises(ValueError, match="mni_affine"):
        model([image], [torch.ones_like(image)], spacing=[(1., 1., 1.)],
              num_visible=16, num_predict=16, num_samples=2)
    with pytest.raises(ValueError, match="two independent views"):
        model.apply_regions(torch.randn(1, 16, 24), torch.ones(1, 16, 21), 1, 1)


def test_base_initialization_is_preserved():
    torch.manual_seed(7338)
    baseline = make_model("none")
    torch.manual_seed(7338)
    adapted = make_model("regions")
    # make_model manually randomizes decoder head after the extra module exists;
    # compare the actual baseline encoder weights initialized before that step.
    for key, value in baseline.state_dict().items():
        if key.startswith(("blocks.", "patch_embed.", "pos_embed.")):
            torch.testing.assert_close(value, adapted.state_dict()[key], rtol=0, atol=0)
