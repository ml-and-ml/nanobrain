# Ryan integration and launch audit

Audit date: 2026-11-09. Full runs are still training; this is an implementation/compatibility audit, not benchmark results.

## Frozen source

| Experiment | Commit | Full Slurm job |
| --- | --- | --- |
| Ryan regions | `f65cb9583236245ec54efd0b2c7e9be7f1a9439b` | 581784 |
| Ryan consistency | `770c81ab6acb7a18c3149003542789667d08063a` | 581785 |
| Ryan coverage | `2f3db395ab71904ce2d7d2d900a7ac629d877a9d` | 581787 |

Each branch preserves Ryan's effective winning config, including FINO age, mirror-level targets, nuisance decoder, seed, native resolution, visible/target budgets and learning-rate schedule. Only the selected regional intervention is active. New region modules are constructed after baseline initialization inside an RNG fork. Region queries use four heads, matching the main-family experiment. Automatic Labless submission remains disabled.

## Validation completed

- Five CPU integration tests passed after the final shared FP32 pooling/statistics update. They cover all three modes; mixed 1-mm and 2-mm native spacing; finite forward/backward and nonzero encoder/query/feedback gradients; distinct visible/target patch indices; crop/pad coordinate mapping; required registration and paired-view guards; and exact preservation of seeded baseline encoder parameters.
- New `apply_regions` paths for both regions and consistency passed `torch.compile(..., backend="aot_eager", fullgraph=True)`: compiled outputs and input gradients match eager. No graph breaks were needed. Coverage sampling and variable native geometry deliberately remain eager; baseline encoder/decoder compilation is retained.
- Slurm smoke job 581779 completed 20 full-shape GPU training steps for all three Ryan experiments. All three saved checkpoints reloaded and returned finite global `[768]` and dense `[24,30,24,768]` embeddings. The region model additionally returned 21 named vectors, including bilateral hippocampus, with 21/21 regions available on the checked scan.
- Full compiled GPU training subsequently passed initial compilation and hundreds of updates in every variant, without exceptions, nonfinite losses/gradients, or increasing memory peaks. Logs are in `campaign/logs/ryan-*-58178*.out`.

## Geometry and evaluation compatibility

Native patch centres use actual voxel patch sizes: `(grid_index + 0.5) * patch_size - 0.5`. Per-image voxel-to-world affines include the precise `src.start - dst.start` crop/pad/jitter translation. Existing world-to-MNI transforms are preserved. Atlas probabilities are multiplied by patch brain-mask fraction. This is an approximate population prior, not an individual anatomical segmentation.

The pinned Brainmarks dataset loader supplies each scan's MNI transform; the probe code reads the standard global/dense output keys and safely ignores additional named-region outputs. Dense affines retain the evaluator's input-voxel-to-world convention. Task-specific preprocessing is composed before model preprocessing. The PMG crop preserves MNI metadata in source; PMG is outside this 11-task open run.

An actual dev sample from each open task passed the model's CPU `transform`: nonempty masked values, finite post-crop affine, and exact preservation of the original finite 4-by-4 MNI transform. Tasks checked: `abide_diagnosis`, `adhd_diagnosis`, `cnp_diagnosis`, `ucsf_idh`, `upenn_survival_1y`, `ixi_age`, `openbhb_age`, `brats_enhancing_tumor`, `soop_lesion`, `soop_mrs_poor`, and `brats_whole_tumor`. The normal ADHD loader message dropped one sample lacking derivatives; this did not fail evaluation preparation. This complements `campaign/evaluation_audit.md`'s complete metadata audit.

## Early observations and limits

At smoke step 20, the fraction of regions with at least one weighted patch equivalent was 0.6071 for standard random sampling (both region-context and consistency runs), versus 0.9747 for coverage sampling on the same image sequence. The visible budget remained 512 and reconstruction target budget 768. This validates the intended sampling effect; it does not establish downstream accuracy.

A later full-training log snapshot showed approximately 168 images/s for region context (step 1400), 168 images/s for consistency (step 1100), and 131 images/s for coverage (step 600), with stable memory around 24–26 GB. Coverage is approximately 22% slower at this snapshot; its eager per-region, per-spacing sampling loop adds launch overhead. These are live observations on different batches, not a controlled throughput benchmark. Source remains frozen; no optimization or hyperparameter changes were introduced after launch.

Regional consistency was active and finite (raw loss about 0.190 at the noted step, outer weight 0.05). Training losses are not comparable measures of downstream quality across objectives. The final 11-task open benchmark scores are required for judging all six proposals.
