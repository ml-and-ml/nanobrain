# Ryan: regions

Parent: pinned Ryan fbc1-s7338 (`60df47644a9a74736daefa6ef9adbd4ea240dc8b` source; see root research record). Full effective recipe is `experiment.yaml`; all baseline hyperparameters, FINO age, mirror targets, nuisance decoder and seed 7338 are retained. This worktree enables exactly one intervention: **regions**.

- `regions`: learned Harvard–Oxford ROI queries attend only to their atlas support, exchange region/global context, and feed a gated residual into patch tokens. The altered tokens feed reconstruction, FINO age, whole-brain pooling and dense inference.
- `consistency`: same-ROI two-mask invariance plus across-subject variance/covariance penalties; outer weight 0.05. No learned ROI head or coverage changes.
- `coverage`: reserve at most 25% of the 512 visible patches for region coverage, then sample disjoint random reconstruction targets. No new learned head or loss.

Each mode uses the frozen 21-region Harvard–Oxford subcortical probability atlas and each scan's existing world-to-MNI registration. Probabilities are population anatomical priors, not a subject segmentation. No pretrained segmentation model or benchmark label is used. Native voxel centres and affines include crop, pad and grid jitter transformations. Missing registration raises rather than inventing anatomy. Atlas buffers are saved in each checkpoint.

`model.transform(sample)` followed by `model.forward_embeddings([transformed])` returns `region_names`, `region_embeddings` (21 × 768), and `region_available` alongside the normal brainmarks global/dense outputs. Availability must be checked before using a regional vector. Only the regions variant learns extra ROI queries; the other variants return atlas-weighted contextual encoder pooling.

No submission is authorized. Automatic training-script submission is disabled and `scripts/submit.py` has an unconditional review guard.

Validation: five CPU integration tests pass for mixed native spacing (1 and 2 mm), all three modes, forward/backward finite gradients, query/feedback gradients, distinct visible/target indices, crop/pad geometry, required registration/two-view guards, and preservation of seeded encoder initialization. Run with `PYTHONPATH=src python -m pytest -q tests/test_region_integration.py`. GPU training/evaluation records are tracked by the root campaign runner.
