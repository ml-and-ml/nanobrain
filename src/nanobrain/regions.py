"""Anatomically grounded region operations shared by the six local experiments.

Weights are atlas probabilities in [0,1], shaped (batch, patches, regions).
An unavailable region always has a zero output plus an explicit availability mask.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def weighted_region_pool(tokens: Tensor, weights: Tensor, min_mass: float = 0.25):
    weights = weights.float().clamp_min(0)
    mass = weights.sum(dim=1)
    available = mass >= min_mass
    pooled = torch.einsum("bnr,bnd->brd", weights, tokens.float())
    pooled = pooled / mass.clamp_min(1e-6).unsqueeze(-1)
    return pooled * available.unsqueeze(-1), available


class RegionContext(nn.Module):
    """Named local queries, brain context, and residual feedback to patch tokens.

    Local attention is exactly restricted to atlas support. A query with no
    support can use finite placeholder attention internally, but its returned
    embedding and all contributions to feedback are masked out.
    """

    def __init__(self, embed_dim: int, num_regions: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.region_identity = nn.Parameter(torch.empty(1, num_regions, embed_dim))
        self.local_norm = nn.LayerNorm(embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.local_out = nn.Linear(embed_dim, embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.context_qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.context_out = nn.Linear(embed_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.feedback = nn.Linear(embed_dim, embed_dim)
        self.feedback_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.region_identity, std=0.02)

    def forward(self, tokens: Tensor, weights: Tensor):
        batch, patches, dim = tokens.shape
        heads = self.num_heads
        pooled, available = weighted_region_pool(tokens, weights)
        regions = pooled.to(tokens.dtype) + self.region_identity
        source = self.local_norm(tokens)
        q = self.query(regions).reshape(batch, -1, heads, dim // heads).transpose(1, 2)
        k = self.key(source).reshape(batch, patches, heads, dim // heads).transpose(1, 2)
        v = self.value(source).reshape(batch, patches, heads, dim // heads).transpose(1, 2)
        support = weights.transpose(1, 2).float()
        bias = support.clamp_min(1e-8).log().masked_fill(support <= 0, float("-inf"))
        bias = torch.where(available.unsqueeze(-1), bias, torch.zeros_like(bias))
        # Explicit attention supports arbitrary atlas masks under every backend,
        # including evaluation contexts that restrict SDPA to Flash/Efficient.
        logits = (q @ k.transpose(-1, -2)).float() / math.sqrt(dim // heads)
        attention = (logits + bias[:, None]).softmax(dim=-1).to(v.dtype)
        local = attention @ v
        regions = regions + self.local_out(local.transpose(1, 2).reshape(batch, -1, dim))
        regions = regions * available.unsqueeze(-1)

        # A real global brain token provides context even if no atlas regions overlap.
        context = torch.cat([regions, tokens.mean(dim=1, keepdim=True)], dim=1)
        qkv = self.context_qkv(self.context_norm(context))
        qkv = qkv.reshape(batch, -1, 3, heads, dim // heads).permute(2, 0, 3, 1, 4)
        key_available = F.pad(available, (0, 1), value=True)
        logits = (qkv[0] @ qkv[1].transpose(-1, -2)).float() / math.sqrt(dim // heads)
        logits = logits.masked_fill(~key_available[:, None, None, :], float("-inf"))
        context_update = logits.softmax(dim=-1).to(qkv[2].dtype) @ qkv[2]
        context = context + self.context_out(context_update.transpose(1, 2).reshape(batch, -1, dim))
        regions = self.output_norm(context[:, :-1]) * available.unsqueeze(-1)

        # Atlas-local feedback plus the mean available-region context. Unassigned
        # brain patches receive the latter; absent ROIs never contribute.
        local_weights = weights.to(regions.dtype) * available[:, None, :]
        local_weights = local_weights / local_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        local_feedback = torch.einsum("bnr,brd->bnd", local_weights, regions)
        global_feedback = regions.sum(dim=1, keepdim=True) / available.sum(dim=1).clamp_min(1)[:, None, None]
        update = self.feedback(0.5 * (local_feedback + global_feedback))
        # Remove projection bias as well when the entire atlas is unavailable.
        update = update * available.any(dim=1)[:, None, None]
        updated = tokens + self.feedback_scale.tanh() * update
        return updated, regions, available


def region_consistency_loss(
    tokens: Tensor,
    weights: Tensor,
    batch_size: int,
    num_samples: int,
    min_mass: float = 1.0,
) -> Tensor:
    """Same-ROI masked-view VICReg; variance is across subjects within each ROI.

    No trainable queries or projection head are used. Covariance uses the batch
    Gram identity, avoiding an R x D x D tensor at full ViT width. One view is a
    diagnostic no-op; actual consistency training requires at least two views.
    """
    if num_samples < 2:
        return tokens.float().sum() * 0
    pooled, available = weighted_region_pool(tokens, weights, min_mass)
    pooled = pooled.reshape(batch_size, num_samples, pooled.shape[1], pooled.shape[2])
    available = available.reshape(batch_size, num_samples, -1)
    losses = []
    for other in range(1, num_samples):
        x, y = pooled[:, 0], pooled[:, other]
        valid = available[:, 0] & available[:, other]
        valid_float = valid.float()
        invariance = ((x - y).square().mean(dim=-1) * valid_float).sum() / valid_float.sum().clamp_min(1)
        count = valid_float.sum(dim=0)
        enough = count >= 2
        variance = x.new_zeros(())
        covariance = x.new_zeros(())
        for view in (x, y):
            mean = (view * valid_float[..., None]).sum(dim=0) / count.clamp_min(1)[:, None]
            centered = (view - mean) * valid_float[..., None]
            var = centered.square().sum(dim=0) / (count - 1).clamp_min(1)[:, None]
            std_penalty = F.relu(1.0 - (var + 1e-4).sqrt()).mean(dim=-1)
            variance = variance + (std_penalty * enough).sum() / enough.sum().clamp_min(1)
            by_region = centered.transpose(0, 1)
            gram = by_region @ by_region.transpose(1, 2)
            cov_norm = gram.square().sum(dim=(-1, -2)) / (count - 1).clamp_min(1).square()
            off_diagonal = (cov_norm - var.square().sum(dim=-1)).clamp_min(0) / view.shape[-1]
            covariance = covariance + (off_diagonal * enough).sum() / enough.sum().clamp_min(1)
        losses.append(invariance + 0.05 * variance + 0.005 * covariance)
    return torch.stack(losses).mean()


def sample_region_sequences(
    mask_patches: Tensor,
    weights: Tensor,
    num_samples: int,
    num_visible: int,
    seq_length: int | None = None,
    min_mask_frac: float = 0.25,
    coverage_fraction: float = 0.25,
) -> Tensor:
    """Reserve visible capacity for named regions, then sample disjoint targets.

    Quotas are capped by a quarter of the original visible budget. Regions with
    no eligible atlas support contribute nothing; the remainder is random brain
    sampling. Targets are sampled uniformly without any region priority.
    """
    batch, patches, _ = mask_patches.shape
    eligible = mask_patches.float().mean(dim=-1) >= min_mask_frac
    if seq_length is None:
        seq_length = int(eligible.sum(dim=1).max().item())
    if not 0 <= num_visible <= seq_length <= patches:
        raise ValueError("Require 0 <= num_visible <= seq_length <= number of patches")
    budget = min(num_visible, max(0, int(num_visible * coverage_fraction)))
    random_order = torch.rand(batch, num_samples, patches, device=mask_patches.device)
    reserved = torch.zeros_like(random_order, dtype=torch.bool)
    num_regions = weights.shape[-1]
    if budget and num_regions:
        per_region = min(patches, math.ceil(budget / num_regions))
        for region in range(num_regions):
            confidence = weights[:, None, :, region].float()
            supported = eligible[:, None] & (confidence >= 0.05)
            priority = -torch.log(random_order.clamp_min(1e-7)) / confidence.clamp_min(1e-7)
            priority = priority.masked_fill(~supported, float("inf"))
            selected = priority.topk(per_region, dim=-1, largest=False).indices
            selected_valid = supported.expand(-1, num_samples, -1).gather(2, selected)
            region_reserved = torch.zeros_like(reserved).scatter(2, selected, selected_valid)
            reserved = reserved | region_reserved
        # Overlap can reduce the reservation count; excess capacity is random.
        # A fresh draw avoids favoring larger regions when trimming to the cap.
        trim_score = torch.rand_like(random_order).masked_fill(~reserved, 2.0)
        trim_ids = trim_score.topk(budget, dim=-1, largest=False).indices
        trim_valid = reserved.gather(2, trim_ids)
        reserved = torch.zeros_like(reserved).scatter(2, trim_ids, trim_valid)
    visible_scores = random_order + (~eligible[:, None]).float() * 4 - reserved.float() * 2
    visible = visible_scores.topk(num_visible, dim=-1, largest=False).indices
    is_visible = torch.zeros_like(reserved).scatter(2, visible, True)
    target_scores = torch.rand_like(random_order) + (~eligible[:, None]).float() * 2
    target_scores = target_scores.masked_fill(is_visible, float("inf"))
    target = target_scores.topk(seq_length - num_visible, dim=-1, largest=False).indices
    return torch.cat([visible, target], dim=-1)
