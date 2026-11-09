# Nanobrain: live leaderboard audit and regional embedding proposal

Checked 2026-11-09, approximately 05:30 UTC. This is a source-code review and experimental proposal; no training or benchmark reproduction was performed.

## Checkout and evidence

- Working repository: `/admin/home/hm/repos/nanobrain`.
- Research branch: `research/labless-2026-11-09`, based on upstream `440fc92a0f949c5874db64ec6491f397bcbda37c`.
- Pinned leading submission: local branch `leader/fbc1-s7338`, commit `60df47644a9a74736daefa6ef9adbd4ea240dc8b`, remote `leader` = `https://github.com/RyanKim17920/nanobrain.git`.
- `leaderboard.json`: 200 public run summaries from the fully paginated experiment ledger, including status and source URLs.
- `fbc1-s7338.yaml`: actual submitted run configuration.
- `source-verification.json`: all 19 submitted file hashes verified; all 18 source/config files present in Git exactly match the pinned commit. The remaining file is the effective run configuration.

## Current leader

| Meaning | Run | Contributor | Open score | Status |
|---|---|---|---:|---|
| Highest submitted score | [fbc1-s7338](https://labless.dev/runs/run_sub_8ce2263544) | RyanKim17920 | 0.5494727418 | Unvalidated |
| Official leader/main flag | [main-1h-20261106](https://labless.dev/runs/run_sub_d5f799669c) | PaulScotti | 0.4727329932 | Main baseline |
| External reference | Walnut | — | 0.556 | Frozen reference, outside the one-hour submitted-run competition |

The highest submission arrived at 2026-11-09 04:43:54 UTC. `open_score` is the arithmetic mean of the fixed 11 task scores, not classification accuracy. Its advantage over main is 0.07673975; its advantage over the prior `fbmirror-s7338` is only 0.00116285. These RyanKim variants use one seed, 7338. The last addition improves the mean but lowers ABIDE, ADHD and the segmentation aggregate relative to `fbmirror`; it does not establish a robust, across-task improvement.

Primary records: [project metadata](https://api.labless.dev/api/nano-projects/nanobrain-smri), [experiment ledger](https://api.labless.dev/api/nano-projects/nanobrain-smri/experiment-log?limit=100), [run detail](https://api.labless.dev/api/runs/run_sub_8ce2263544), [submitted source](https://api.labless.dev/api/runs/run_sub_8ce2263544/source). Pagination used `after_updated_at` and `after_run_id`; no alternative time track was found.

## How fbc1 works

It is a 3D masked autoencoder built around a ViT-B: 12 encoder layers, width 768, 12 attention heads, and a four-layer, width-512 decoder. It has no active class token. Each image produces two masked sequences, each containing 512 visible patches and 768 reconstruction targets. Patch support is approximately 8 mm per side. Training uses native voxel spacing through spacing-aware patch embedding/reconstruction kernels, while the submitted inference path still resamples to 1 mm.

The objectives are:

1. Reconstruct hidden brain intensities with a brain-mask-weighted squared-error loss.
2. Predict local intensity-threshold occupancy at 2 mm subcells from visible encoder tokens. These are self-generated intensity targets, not anatomical labels.
3. Predict unusually high local intensities relative to a mirrored contralateral neighborhood. This is an asymmetry target, not an assumption that both hemispheres must match. The mirror uses the centered box, not registered anatomical correspondence.
4. Predict age from pooled visible encoder features through a small FINO auxiliary head. Age is the only enabled FINO target in this run. Its loss weight is 0.03; `age: 3.0` scales the gradient reaching the backbone and is distinct from the loss coefficient. The code excludes a collection's metadata because it reports overlap with OpenBHB.

Manufacturer, field strength, contrast, voxel-spacing bucket and collection embeddings are summed into decoder tokens during training. This gives the reconstruction decoder acquisition information directly; encouraging the encoder to retain less scanner style is the intent, not a guaranteed invariance property. These metadata and auxiliary heads are unnecessary at inference.

Muon optimization, FlashAttention-3, compiled operations, reduced decoder work and data preparation changes support the time budget. The run reports 25,728 steps, 823,296 image presentations and 3,572.88 seconds on one H100, with 104,864,513 training parameters. Presentations are not unique subjects or unique scans.

Inference retains the encoder and returns the mean of final patch features as a 768-dimensional global embedding. It can also return the spatial grid of 768-dimensional patch embeddings. Consequently, much of the machinery for regional embeddings already exists.

Pinned source pointers: [model and inference](https://github.com/RyanKim17920/nanobrain/blob/60df47644a9a74736daefa6ef9adbd4ea240dc8b/src/nanobrain/model.py#L580), [FINO](https://github.com/RyanKim17920/nanobrain/blob/60df47644a9a74736daefa6ef9adbd4ea240dc8b/src/nanobrain/fino.py#L100), [nuisance decoder](https://github.com/RyanKim17920/nanobrain/blob/60df47644a9a74736daefa6ef9adbd4ea240dc8b/src/nanobrain/nuisdec.py).

## Obtaining named anatomy

Two useful routes:

- **Off-the-shelf segmentation:** [SynthSeg](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg) produces anatomical labels across varied contrasts and resolutions. Its [label table](https://github.com/BBillot/SynthSeg/blob/master/data/labels%20table.txt) includes left hippocampus 17 and right hippocampus 53, as well as amygdala, thalamus, ventricles and other structures. Keep its output affine: it internally resamples to 1 mm, so its output indices need not coincide with the input grid. This is a convenient research prototype route, not automatically competition eligible.
- **Atlas registration:** register a subject scan to a template and map a labeled/probabilistic atlas into the subject's coordinates. The [FSL registration practical](https://pages.fmrib.ox.ac.uk/fslcourse/practicals/registration/index.html) explicitly demonstrates a Harvard–Oxford left hippocampus ROI and transforming masks between spaces. Classical registration avoids loading a pretrained MRI neural network. Atlas-derived regions are uncertain estimates rather than subject-specific ground-truth segmentations. Nonlinear registration improves flexibility but brings extra compute and failure modes; affine-only alignment is a coarse baseline.

For hard labels, use label-preserving interpolation; for probability maps, interpolate probabilities then validate range and normalization. [ANTs documents transforms and label interpolation](https://antspy.readthedocs.io/en/stable/registration.html). Inspect overlay alignment, left/right orientation and registration failures before deriving embeddings.

Current nanobrain RAS orientation, crop/pad and physical coordinates do **not** provide MNI registration or named regions. The native training loader currently discards the full affine before the forward pass. Carry the ROI maps through precisely the same transforms and patch sampling, or carry the affine and atlas transform needed to generate them. The returned dense embedding has patch stride eight; do not mistake a dense-grid index for a one-voxel index.

The [challenge rules](https://github.com/MedARC-AI/nanobrain/blob/440fc92a0f949c5874db64ec6491f397bcbda37c/README.md#rules) allow one H100-hour, FOMO300K pretraining images, and no brain-MRI pretrained weights or teachers. SynthSeg-derived training targets should not be assumed eligible, including on the argument that SynthSeg uses synthetic-image training. Atlas priors and registration preprocessing accounting are not explicitly resolved in that README. Resolve those questions before an official submission; the architecture proposal itself does not require external pretrained weights.

## Three proposed improvements

### 1. Anatomically grounded contextual region tokens

Start with 16–32 coarse regions, retaining separate left/right regions. The name is an index in a learned embedding table, not a text prompt: calling an unconstrained token “hippocampus” does not make it attend to the hippocampus.

Let `H[i]` be globally contextualized encoder patch features and `a[r,i]` the soft fraction/probability of region `r` in patch `i`. First obtain a no-training baseline:

`local[r] = sum_i a[r,i] * H[i] / (sum_i a[r,i] + epsilon)`.

Then initialize `q[r] = region_id_embedding[r] + local[r]`; use one cross-attention layer from these queries to `H`, with a soft `log(a[r,i] + epsilon)` attention bias. Add a small region/global attention block so that the left hippocampus representation can incorporate ventricular, cortical and contralateral context. Preserve a residual local path so the result does not become merely a whole-brain vector with a region name.

Expose `[batch, region, 768]` embeddings plus region names, coverage and availability. Preserve the existing global/dense embedding interface for Brainmarks; expose named outputs separately. The existing patch tokens already have whole-image context, so atlas-weighted pooling is a meaningful comparator, not a context-free baseline.

Train the head jointly using region reconstruction/level-set targets and proposal 2. If the region outputs are intended to improve the official global score, make the regional auxiliary losses update the encoder and evaluate the original global readout; do not silently replace the benchmark protocol. Compare ordinary mean pooling, atlas-weighted pooling, learned local queries, and local-plus-context queries. Include shuffled-region or matched-coordinate controls to test whether anatomy contributes beyond location.

### 2. Same-region consistency across the two existing masked views

The leader already encodes two independently sampled views per image but does not directly align their regional embeddings. Add a modest consistency objective for matching regions across those views, using only regions with enough visible evidence in both. This can encourage stable regional identity when different local patches are hidden, without a third backbone pass or external teacher.

Use a small projector with a variance/covariance anti-collapse term, alongside the existing reconstruction and grounded regional losses; naive vector matching alone can collapse. Start with the same scan/contrast and independent masks. Do not force left and right hippocampi to have identical features, and do not initially impose equality across contrasts that carry different pathology information.

Ablate consistency off/on at fixed one-hour compute. Measure view-to-view stability, regional variance, anatomical specificity, and all 11 task scores. Strong invariance may erase useful local differences; improve stability only when task information is preserved.

### 3. Region-balanced visibility and targets, then limited finer patches

Uniform patch sampling gives large structures many opportunities and can leave a small hippocampus with little visible evidence. Reserve a small part of the existing 512-visible budget for rotating, confidently mapped regions; leave most visible tokens sampled globally and initially keep the 768 disjoint prediction targets random. This isolates a sampling change from adding region-query parameters. Region-balanced targets/losses are a later, separate ablation, with caps so tiny/noisy ROIs cannot dominate.

First test balancing with the existing 8 mm patches. If localization remains the limiting factor, replace a bounded subset of coarse tokens with approximately 4 mm patches around selected small regions, using an explicit scale embedding and a compatible patch projection. Do not switch the whole volume to 4 mm patches: at fixed volume that increases token count roughly eightfold. Prefer rotating region subsets so every region need not be present in every training sequence.

Ablate uniform versus balanced sampling before adding finer patches. Track ROI coverage, examples/hour, boundary sensitivity and the official mean. A finer branch helps only if local information gained compensates for lost training throughput. The current model is not a hippocampal-subfield delineator, and coarse atlas membership should not be presented as one.

## Suggested first experiment

Validate anatomy mapping on a small FOMO-only development set, then compare atlas-weighted dense-feature pooling with the global feature, using a subject-disjoint development evaluation. Add the late region head only after that mapping works. Run balanced sampling and regional consistency as separate arms before combining them. Repeat promising variants across seeds: a 0.00116 leaderboard gap is too small to treat as a reliable gain without replication. Leave the fixed public evaluator unchanged, report per-task tradeoffs, and record preprocessing cost separately rather than assuming it is free.
