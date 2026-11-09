# Main / regions

A learned query per Harvard–Oxford label attends only to patch support for that label, then exchanges information with the other available regions and a whole-brain token. A residual feedback projection updates every patch embedding, so both reconstruction training and the standard global/dense Brainmarks outputs use the region context. `forward_region_embeddings` exposes the 21 named contextual embeddings and an availability mask.

This is an independent arm from the pinned upstream main model. The original full training file list, ViT-B width (768), random seed (7338), and 60-minute wall-clock budget are preserved. New query parameters are initialized after the original network inside an isolated RNG context, preserving backbone initialization and the subsequent mask RNG state.

Anatomy uses the dataset-provided native-world-to-MNI152NLin6Asym affine composed with the actual cropped/resampled/padded image affine. Eight within-patch samples approximate overlap with the fixed Harvard–Oxford probability atlas. These are population atlas memberships, not subject-specific segmentations. Missing transforms raise; no coordinate bins are assigned anatomical names. Atlas buffers are embedded in checkpoints.

Run `python -m nanobrain.train --config experiment.yaml` with the campaign environment. The root campaign runner controls output directories and full local Brainmarks evaluation. Labless submission is disabled pending user review.

Validation lives in `tests/test_region_integration.py` and the campaign's `shared/test_regions.py`. The tests check anatomical masking, feedback gradients, missing-region behavior, anti-collapse behavior, disjoint sampling, original backbone/RNG initialization, a real hippocampal atlas lookup, and checkpoint/named-output round trips. Reduced models in tests are diagnostics only; campaign training uses the full baseline configuration.
