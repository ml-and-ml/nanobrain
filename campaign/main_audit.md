# Main experiment audit

This audit reviews the three independent upstream-main arms against base commit `440fc92a0f949c5874db64ec6491f397bcbda37c`. It does not alter the frozen training worktrees or their source snapshots.

## Isolation and comparability

| Arm | Commit | Trainable parameters | Added behavior |
|---|---|---:|---|
| main-regions | `46715080735c5dda93ef2827f33296866d02d321` | 104,153,345 | Learned anatomical queries, contextual exchange, and feedback into encoder patch features |
| main-consistency | `130d1aabf04754bcbc39d9a37ef94df86a04c2a0` | 98,817,280 | Same-region masked-view consistency plus variance/covariance regularization |
| main-coverage | `695148c74e9093554c0895c9d8b39820f8583791` | 98,817,280 | Anatomical quotas within the existing visible-patch budget |

Only the regions arm adds trainable parameters: 5,336,065 above the original model. Consistency introduces no query/projection head; coverage introduces neither a head nor a new loss. Their source implementation is shared, with only the selected mode active.

Configuration comparison against the pinned main recipe adds only `region_mode`, `region_consistency_weight=0.05`, and `region_coverage_fraction=0.25`. All three retain the full 86,713-image file list, 768-dimensional ViT-B encoder, batch size 32, two masks per image, 640 visible patches, 1,920 prediction targets, seed 7338, original optimizer and learning-rate schedule, and 60-minute wall-clock limit. Atlas preprocessing and compilation consume that time budget. Therefore step counts can differ across arms under the fixed wall budget.

Tests verify exact equality of every original backbone/decoder state tensor and the post-construction CPU RNG state between the original initialization path and each arm. New query parameters are initialized after the original model inside `torch.random.fork_rng(devices=[])`. The atlas constructor consumes no model initialization randomness.

The consistency arm's logged training loss includes its auxiliary term. **Do not directly rank its raw training loss against the reconstruction losses of the other two arms.** Compare the complete, identical Brainmarks evaluation instead. Historical leaderboard reference scores are not newly rerun controls, and this single-seed campaign does not establish statistical significance.

## Anatomical and inference integration

The training loader preserves `mni_affine`. Preprocessing retains the actual final voxel-to-native-world affine after orientation, spacing changes, and crop/pad. Patch centers are converted from patch indices to voxel centers before composing that affine with the native-world-to-MNI transform. Atlas probabilities are weighted by brain-mask coverage; absent regions have explicit availability masks and zero returned region embeddings.

For regions, the same contextual head updates patch embeddings in training and inference. Standard global features average the updated patch embeddings, and dense features contain those same updated embeddings. Named embeddings are exposed by `forward_region_embeddings` without changing the locked evaluator. Consistency and coverage retain the original inference encoder.

All three full-width GPU smoke tests completed 20 optimizer steps on real FOMO inputs, saved checkpoints, and passed real-input global/dense output checks. The regions checkpoint additionally produced 21 finite named region embeddings, with all 21 available for the checked scan. Unit tests verify query/feedback gradients, exact local anatomical attention constraints, missing-region fallback, anti-collapse behavior, disjoint sampling, a real left-hippocampus atlas lookup, and checkpoint restoration.

## Compilation and plotting

All three full jobs successfully compiled and advanced through at least 5,000 optimizer steps without a reported compilation failure. The fixed-length training path adds no host `.item()` synchronization; atlas queries run outside the compiled model. Regional attention uses explicit short-query matrix products, avoiding unsupported masked FlashAttention kernels. Pooling and covariance statistics explicitly use float32.

The plot call passes the current batch's anatomical weights, sliced to its eight examples. `plot_examples` uses `torch.compiler.set_stance("force_eager")`, one masked view, and all remaining targets. The consistency term becomes a diagnostic no-op for that single view; coverage still samples distinct visible and target patches. The model returns to training mode afterward.

An end-to-end CPU diagnostic invoked the actual `plot_examples` function for all three modes, with full remaining-target decoding and explicit region weights. Every mode produced three nonempty PNGs and restored training mode. Diagnostics used temporary directories and small models; the campaign jobs retain the full configurations.

Actual GPU step-5,000 plot cycles are verified for all three arms. Each wrote sagittal, coronal, and axial PNGs, saved its checkpoint, and resumed compiled training. Observation time: 2026-11-09T06:12:43.545401+00:00.

| Arm | Latest logged step at verification | Nonempty plots | PNG size range (bytes) |
|---|---:|---:|---:|
| main-regions | 5,800 | 3 | 259,244–383,983 |
| main-consistency | 5,500 | 3 | 258,297–383,033 |
| main-coverage | 5,200 | 3 | 257,557–380,344 |

Artifacts are under `/data/hassan/nanobrain/anatomical-regions-20261109/<arm>/plots/mae_{sag,cor,ax}_0005000.png`. This closes the smoke-test gap caused by its `compile=false` and `plot_period=0` overrides. Full evaluation scores were still pending when this audit was written.

## Runner review

`run.py` rejects dirty worktrees, verifies the pinned evaluator installation, records source/config identities, and checks completed training before evaluation. The fixed evaluator emits the `open` aggregate only when all required tasks are present; the campaign verifies that aggregate before reporting completion. Checkpoint diagnostics reconstruct the real image using its stored scale and affines. Neither the campaign runner nor the branch launchers submit to Labless.

Operational note: the four-hour job allocation includes training plus evaluation. If an evaluation exceeds the remaining allocation, it requires a monitored evaluation-only retry; Slurm's `--requeue` flag alone does not guarantee automatic retry after a wall-time limit.
