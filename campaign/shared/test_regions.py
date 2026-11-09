"""Scientific-behavior checks for the independent region experiments."""

import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("region_operations", Path(__file__).with_name("regions.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RegionContext = module.RegionContext
weighted_region_pool = module.weighted_region_pool
region_consistency_loss = module.region_consistency_loss
sample_region_sequences = module.sample_region_sequences


def test_pool_respects_anatomy_and_reports_missing_regions():
    tokens = torch.tensor([[[2.0, 4.0], [6.0, 8.0], [100.0, 100.0]]])
    weights = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    pooled, available = weighted_region_pool(tokens, weights)
    assert torch.equal(pooled[0, 0], torch.tensor([4.0, 6.0]))
    assert torch.equal(pooled[0, 1], torch.tensor([100.0, 100.0]))
    assert available.tolist() == [[True, True, False]]
    assert not pooled[0, 2].any()


def test_region_context_local_mask_feedback_and_gradients():
    torch.manual_seed(17)
    context = RegionContext(16, 3)
    tokens = torch.randn(2, 8, 16, requires_grad=True)
    weights = torch.zeros(2, 8, 3)
    weights[:, :3, 0] = 1
    weights[:, 3:6, 1] = 1
    captured = []
    hook = context.local_out.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    updated, regions, available = context(tokens, weights)
    changed = tokens.detach().clone()
    changed[:, 7] = torch.randn(2, 16) * 20
    context(changed, weights)
    hook.remove()
    # Background affects subsequent whole-brain context but cannot enter local attention.
    torch.testing.assert_close(captured[0][:, :2], captured[1][:, :2])
    assert updated.shape == tokens.shape and regions.shape == (2, 3, 16)
    assert not torch.allclose(updated, tokens)
    assert not available[:, 2].any() and not regions[:, 2].any()
    (updated.square().mean() + regions[:, :2, 0].mean()).backward()
    for name in ("region_identity", "query.weight", "feedback.weight"):
        grad = dict(context.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    fallback, unavailable, present = context(tokens.detach(), torch.zeros_like(weights))
    torch.testing.assert_close(fallback, tokens.detach())
    assert not unavailable.any() and not present.any()
    other, _, _ = context(tokens.detach(), weights.roll(3, dims=1))
    assert not torch.allclose(updated.detach(), other)


def test_consistency_detects_collapse_and_disagreement():
    # Four orthogonal, centered subject features, scaled to unit unbiased variance.
    signs = torch.tensor([[(-1.0) ** ((i & j).bit_count()) for j in range(1, 5)] for i in range(8)])
    signs = signs * (7 / 8) ** 0.5
    diverse = signs[:, None, None, :].expand(-1, 2, 2, -1).reshape(16, 2, 4).clone()
    weights = torch.ones(16, 2, 2)
    collapsed = torch.zeros_like(diverse, requires_grad=True)
    collapse_loss = region_consistency_loss(collapsed, weights, 8, 2)
    diverse_loss = region_consistency_loss(diverse, weights, 8, 2)
    assert collapse_loss > 0.09
    assert diverse_loss < 1e-5
    disagree = diverse.reshape(8, 2, 2, 4).clone()
    disagree[:, 1] += 2
    assert region_consistency_loss(disagree.flatten(0, 1), weights, 8, 2) > 3.9
    collapse_loss.backward()
    assert torch.isfinite(collapsed.grad).all()
    assert region_consistency_loss(diverse, torch.zeros_like(weights), 8, 2) == 0


def test_coverage_protects_small_regions_without_target_overlap():
    weights = torch.zeros(2, 128, 4)
    for region in range(4):
        weights[:, region, region] = 1
    mask = torch.ones(2, 128, 8)
    torch.manual_seed(33)
    order = sample_region_sequences(mask, weights, 3, 16, 80)
    torch.manual_seed(33)
    assert torch.equal(order, sample_region_sequences(mask, weights, 3, 16, 80))
    assert order.shape == (2, 3, 80)
    for sequence in order.flatten(0, 1):
        assert len(sequence.unique()) == 80
        assert set(range(4)).issubset(set(sequence[:16].tolist()))
    # No support still produces finite, distinct random brain samples.
    fallback = sample_region_sequences(mask, torch.zeros_like(weights), 3, 16, 80)
    assert all(len(row.unique()) == 80 for row in fallback.flatten(0, 1))
