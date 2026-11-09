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

Artifacts are under `/data/hassan/nanobrain/anatomical-regions-20261109/<arm>/plots/mae_{sag,cor,ax}_0005000.png`. This closes the smoke-test gap caused by its `compile=false` and `plot_period=0` overrides. Final evaluation validation and results are recorded below.

## Runner review

`run.py` rejects dirty worktrees, verifies the pinned evaluator installation, records source/config identities, and checks completed training before evaluation. The fixed evaluator emits the `open` aggregate only when all required tasks are present; the campaign verifies that aggregate before reporting completion. Checkpoint diagnostics reconstruct the real image using its stored scale and affines. Neither the campaign runner nor the branch launchers submit to Labless.

Operational note: the four-hour job allocation includes training plus evaluation. If an evaluation exceeds the remaining allocation, it requires a monitored evaluation-only retry; Slurm's `--requeue` flag alone does not guarantee automatic retry after a wall-time limit.


## Final evaluation validation

Completed at 2026-11-09T07:16:17.237137+00:00. All three main arms completed their full wall-budget training and all 11 fixed open tasks. An independent audit verified all **33 task JSON files**: task identity, `subset=dev`, exact intended checkpoint, clean experiment `cwd_git` source SHA, and finite metrics. All three aggregate scores equal the unweighted mean of the 11 finite task scores. Training summaries report `done=true`, no signal termination, and the expected clean source commits. The experiment-config SHA-256 values still match the launch records; the worktrees remain clean.

Sample counts, fold counts, and evaluated subject/session/fold assignments match exactly across the three arms for every task. These identities were compared internally and are not reproduced in this report.

| Arm | Open score | 95% sample-bootstrap interval | Delta vs historical main | Training steps | Reported wall seconds |
|---|---:|---|---:|---:|---:|
| main-regions | 0.486434489 | 0.457609–0.515164 | +0.013701496 | 20,364 | 3606.40 |
| main-consistency | 0.467113572 | 0.437940–0.497189 | -0.005619421 | 20,166 | 3606.52 |
| main-coverage | 0.461272816 | 0.432793–0.490943 | -0.011460177 | 20,443 | 3606.18 |

The learned-regions arm has the highest point estimate among the main variants and leads 9 of 11 individual tasks. Consistency leads OpenBHB age (R² 0.846486 versus 0.813191 for regions), while coverage leads IXI age (R² 0.831960 versus 0.828211 for regions). Regions leads all six classification tasks and all three segmentation tasks. Its enhancing-tumor Dice is 0.617050, versus 0.603514 for consistency and 0.598750 for coverage; SOOP lesion Dice is 0.566313, versus 0.525479 and 0.557092. Whole-tumor Dice is very close between regions and consistency: 0.828624 versus 0.827744, with coverage at 0.823240. These are single-seed task tradeoffs, not evidence of significant superiority.

### Recovered allocator warnings

The main-regions and main-consistency whole-tumor evaluations emitted CUDA allocator allocation warnings at 07:06:27 UTC and 07:10:12 UTC respectively, each while requesting approximately 4.50 GB. Both continued fitting subsequent folds and finished successfully. Read-only inspection of the pinned evaluator's `probes.py` and `logistic.py` found no OOM handler that drops subjects, shrinks the protocol, or changes the probe after an allocation warning. A propagated exception would be caught at the task boundary in `run.py`, mark the task failed, and produce a nonzero evaluator exit; that did not occur.

All three final whole-tumor files contain exactly **500 evaluated subjects, five folds of 100 subjects, five selected hyperparameter records, and 500 finite subject-level Dice scores**. Subject/session/fold assignments match exactly across arms. Thus the recovered warnings did not result in missing evaluated subjects or a changed fixed evaluation protocol. The original deterministic negative-patch subsampling in the pinned logistic probe remains unchanged; it is unrelated to these allocation warnings.

No source edits, evaluation changes, relaunches, or Labless submissions were made during this audit.
