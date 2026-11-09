# Anatomical priors for the six local experiments

The implementation uses the public Harvard–Oxford **subcortical probabilistic atlas**, at 2 mm, with its 21 original region channels. These include separate left/right hippocampus, amygdala, thalamus, caudate, putamen, pallidum, accumbens, lateral ventricle, cerebral cortex and white matter, plus brainstem. These are population anatomical priors. They are not individually validated segmentations or hippocampal subfields.

The important discovery is that FOMO300K and Brainmarks already include the required registration. `/data/brainmarks-smri/fomo300k/README.md` documents `mni_affine` inside each packed NPZ: a classical ANTs 12-parameter affine from the source scan's RAS-world coordinates to **MNI152NLin6Asym**. Brainmarks `ImageInput` passes the same field at evaluation. Existing nanobrain ignored it. This campaign retains it; no additional neural segmentation model or target labels are needed.

The existing pretraining filelist selects scans without QC warnings. Affine-only registration remains approximate, particularly in pathological or very atypical anatomy. Image-mask containment QA cannot establish segmentation accuracy. Benchmarks continue to use their original global/dense interface and unchanged target labels; named regions have a separate output path.

## Data and provenance

- [Nilearn official Harvard–Oxford loader](https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_harvard_oxford.html) identifies the FSL-packaged download.
- Exact archive: `https://www.nitrc.org/frs/download.php/9902/HarvardOxford.tgz`.
- Exact files: `data/atlases/HarvardOxford/HarvardOxford-sub-prob-2mm.nii.gz` and `data/atlases/HarvardOxford-Subcortical.xml`.
- [FSL dataset documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/datasets.html) documents probabilistic volume indexing, anatomical atlas derivation and standard template; use its linked license/attribution requirements.
- [TemplateFlow's original release](https://github.com/templateflow/templateflow/releases/tag/1.0.0) explicitly identifies Harvard–Oxford's native space as **MNI152NLin6Asym**. This matches the Brainmarks registration target, rather than assuming all MNI templates are equivalent.
- SHA256 hashes, affine, label order and the prepared asset hash are recorded in `provenance.json`.
- Source archive/extracted source files and the prepared asset live in `/data/hassan/nanobrain/atlas/`. No patient images are packaged with the atlas.

The XML indices are already the zero-based probability-volume indices. Unlike a hard-label atlas, no `+1` shift is applied. The source image is LAS with affine diagonal `[-2,2,2]`; the full inverse affine handles left/right orientation. The implementation does not assume array index zero is anatomical left.

## Preparing and using the module

Run `python campaign/atlas/prepare_atlas.py` with NumPy and nibabel installed. It writes `/data/hassan/nanobrain/atlas/harvard_oxford_subcortical.npz` (about 434 KB compressed). Download source paths can be overridden with `--download-dir`; output with `--output`.

Copy `atlas.py` into `src/nanobrain/atlas.py`. `AtlasPrior()` reads the asset from the default path, or `NANOBRAIN_ATLAS_PATH`, or its `asset_path` argument. All 21 atlas probability maps (uint8 percentages, approximately 19 MB) and their inverse affine are persistent model buffers. Checkpoints can therefore be restored without the external asset. Calling a fresh model with neither asset nor restored atlas buffers fails explicitly.

```python
atlas = AtlasPrior().to(device)
centres = patch_indices * patch_size + (patch_size - 1) / 2
weights = atlas.patch_weights(centres, image_affine, mni_affine, patch_extent)
# centres: [B,N,3] voxel coordinates on the actual current image grid
# affines: [B,4,4]; patch_extent: [3] or [B,3] actual voxel side lengths
# weights: [B,N,21], approximately mean atlas probability over each patch
```

Mapping is `atlas_voxel = inverse(atlas_affine) @ mni_affine @ image_affine @ image_voxel`. All reorientation, crop, padding, resampling and jitter changes must be included in `image_affine`. Native Ryan patches use their actual native voxel sizes, not the spacing-normalized positional-table coordinates.

Eight equal-volume octant-centre samples approximate the patch overlap; each queries the 2-mm probability atlas with nearest-neighbour interpolation. Averaging the probabilities gives soft weights while keeping the atlas small and frozen. This is an eight-point numerical approximation, not exact voxel overlap. The current implementation preserves overlapping/partial probabilities instead of forcing regions to sum to one. It yields zero outside atlas coverage, zero for a non-finite transform, and rejects missing `mni_affine`. Downstream heads must mark regions with insufficient observed evidence unavailable.

## Validation

`test_atlas.py` tests exact voxel/label correspondence, left/right hippocampus identity, composed world transforms, flip/crop equivariance, out-of-bounds/nonfinite/missing transforms, and checkpoint restoration without an external atlas file. Run with `python campaign/atlas/test_atlas.py` in the campaign Torch environment.

`audit_data.py` inspects only a deterministic subset of the **pretraining** filelist and writes aggregate geometry/brain-mask containment metrics to `data_qa.json`. It writes no individual image, filename or subject identifier. Brain-mask containment is necessary but is insufficient to establish anatomical alignment accuracy.

## Review before any competition submission

These are local research experiments. Public atlas priors are not assumed to be Labless eligible; explicit rule interpretation and accounting must be reviewed before submission. The existing Brainmarks registration is precomputed in the released data, so this implementation performs lookup only. No Labless submission code is called by this module or the atlas preparation/audit tools.
