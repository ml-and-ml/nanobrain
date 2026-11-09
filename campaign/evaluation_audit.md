# Evaluation geometry and atlas provenance audit

Read-only audit, 2026-11-09. Evaluator checkout verified at commit `8ef7115d519a1d1f13910c1e13bcae331b0ca3ad`. Active experiment worktrees were not changed. The checks below concern the public **dev** split used by the campaign, not the held-out test split. Counts are image-task instances; some images occur in multiple tasks.

Evaluator provenance is verified from the installed distribution's `direct_url.json`, which records the pinned source commit. Its task-result `git` field searches upward from the installed package inside this repository's `.venv`, so it reports the enclosing research repository rather than the evaluator source revision. The separate `cwd_git` field correctly records each experiment worktree (with an abbreviated commit hash); final verification matches that against the full training commit. Training seed is 7338 and the unchanged evaluator seed is 0.

## Affine availability across the 11 open tasks

For every selected row, the stored `mni_affine_path` was present in the dataset's shard index and mapped to the same shard as its image. One stored affine per task was also read and checked for shape `(4,4)`, finite entries and positive determinant of its linear part. These checks do not establish anatomical registration accuracy for every image.

| Open task | Dev image-task instances | Missing affine members | Affine/image shard mismatch | Additional task transform |
|---|---:|---:|---:|---|
| abide_diagnosis | 500 | 0 | 0 | None |
| adhd_diagnosis | 254 | 0 | 0 | None |
| cnp_diagnosis | 131 | 0 | 0 | None |
| ucsf_idh | 248 | 0 | 0 | None |
| upenn_survival_1y | 301 | 0 | 0 | None |
| ixi_age | 281 | 0 | 0 | None |
| openbhb_age | 500 | 0 | 0 | None |
| brats_enhancing_tumor | 500 | 0 | 0 | None |
| soop_lesion | 500 | 0 | 0 | None |
| soop_mrs_poor | 368 | 0 | 0 | None |
| brats_whole_tumor | 500 | 0 | 0 | None |
| **Total** | **4,083** | **0** | **0** | |

The evaluator normally drops one ADHD dataset sample without derivatives before task selection. The table reflects the resulting selected samples; no additional filtering was introduced by this audit.

For one selected image per task, a separate header-only check also confirmed identical image/brain-mask spatial shapes and exactly matching NIfTI best affines (maximum absolute matrix difference `0.0` in all 11 cases). This read compressed headers, not voxel arrays or segmentation targets.

## Image geometry and target separation

- [`BrainDataset.__getitem__`, lines 87–108](https://github.com/MedARC-AI/brainmarks-smri/blob/8ef7115d519a1d1f13910c1e13bcae331b0ca3ad/src/brainmarks_smri/datasets.py#L87) constructs the model input using only `image`, `brain_mask` and `mni_affine`. Diagnosis, age and segmentation targets are kept in a separate dictionary. Its optional model transform receives only the image-input dictionary.
- [`compute_embeddings`, lines 284–290](https://github.com/MedARC-AI/brainmarks-smri/blob/8ef7115d519a1d1f13910c1e13bcae331b0ca3ad/src/brainmarks_smri/probes.py#L284) separates inputs and targets, then calls `model.forward_embeddings(image_inputs, return_dense)`. Targets are used subsequently to fit/score the frozen-feature probes. The atlas lookup receives coordinates, image/MNI affine matrices and the fixed atlas; brain-mask coverage gates the weights. It does not receive task targets.
- The [11-task group definition](https://github.com/MedARC-AI/brainmarks-smri/blob/8ef7115d519a1d1f13910c1e13bcae331b0ca3ad/src/brainmarks_smri/tasks.py#L801) and local task objects both show no additional task transform for these tasks. The PMG task, outside this group, masks an anterior/posterior window without changing the grid affine and [explicitly retains `mni_affine`](https://github.com/MedARC-AI/brainmarks-smri/blob/8ef7115d519a1d1f13910c1e13bcae331b0ca3ad/src/brainmarks_smri/tasks.py#L413).
- [BraTS](https://github.com/MedARC-AI/brainmarks-smri/blob/8ef7115d519a1d1f13910c1e13bcae331b0ca3ad/src/brainmarks_smri/datasets.py#L210), [UCSF and UPENN](https://github.com/MedARC-AI/brainmarks-smri/blob/8ef7115d519a1d1f13910c1e13bcae331b0ca3ad/src/brainmarks_smri/datasets.py#L266) use T1w derivatives for already co-registered sequences. [SOOP](https://github.com/MedARC-AI/brainmarks-smri/blob/8ef7115d519a1d1f13910c1e13bcae331b0ca3ad/src/brainmarks_smri/datasets.py#L260) instead uses each modality's own derivatives because its sequences are not co-registered.
- Nanobrain's main and Ryan inference paths retain the input-world-to-MNI matrix while updating the voxel-to-input-world affine through crop, RAS orientation, resampling and padding. Atlas coordinates are composed as `inverse(atlas_affine) @ mni_affine @ current_image_affine`. Ryan native training additionally accounts for its actual voxel patch sizes and crop/pad/jitter offsets. The original dense-feature affine still maps model-input voxels back to image world coordinates, as required by the segmentation probe.

No evaluator augmentation or model-coordinate mismatch was found in this code review. The regional model changes the learned features; it does not change the evaluator's targets, folds or global/dense probe protocol. The atlas/checkpoint geometry tests and the separate 16-image FOMO containment audit are recorded in [`atlas/README.md`](atlas/README.md) and [`atlas/data_qa.json`](atlas/data_qa.json).

## External anatomical supervision and limits

The public Harvard–Oxford subcortical atlas supplies 21 probability channels at 2-mm resolution, including bilateral hippocampus. According to the [FSL source documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/other/datasets.html#harvard-oxford-cortical-and-subcortical-structural-atlases), its population probabilities were derived from semi-automated anatomical segmentations of 37 healthy adults aged 18–50, affine-registered and combined. These experiments therefore use **external anatomical supervision through a fixed atlas**; the anatomical names are not learned solely from FOMO images.

The atlas is natively in MNI152NLin6Asym, matching the [existing Brainmarks ANTs affine-registration target](https://github.com/MedARC-AI/brainmarks-smri/blob/8ef7115d519a1d1f13910c1e13bcae331b0ca3ad/src/brainmarks_smri/utils/preprocessing.py#L26). The new implementation introduces no pretrained neural segmentation teacher and performs no new registration. It reuses the dataset's supplied masks and affine matrices; the dataset's original preprocessing itself used SynthStrip.

The source archive, atlas image, label XML and derived asset hashes are recorded in [`atlas/provenance.json`](atlas/provenance.json). The derived asset SHA256 is `1a2bd7c02dacb2a4c224b075c05fc2a9284a02e8d1c273f733784ee5052b953c`. Atlas arrays and affine are stored in model checkpoints.

Affine alignment, 2-mm population probabilities and eight-point patch integration provide approximate regional localization. They do not validate a particular person's segmentation, hippocampal subfields, pathological deformation or age-specific anatomy. The brain-mask containment audit checks gross geometry only. External-atlas eligibility under Labless's FOMO-only pretraining/no-pretrained-teacher rules remains unresolved; these are local research runs and no submission is authorized before user review.

This document contains aggregate checks and code/data provenance only. No raw images, subject identifiers or task-label values were copied into it.
