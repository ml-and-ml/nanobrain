# Ryan integration and launch audit

Audit date: 2026-11-09. All three Ryan training runs and their full 11-task open evaluations completed. Early observations below are retained as a launch record; final results are appended at the end.

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


## Final results: all three completed

Each run completed all 11 open/dev tasks, covering 4,083 image-task samples. All reported task metrics, aggregate scores and 1,000 aggregate bootstrap replicates are finite. Every task records its expected clean model-source commit. No Labless submission occurred.

| Variant | Open score | 95% bootstrap interval | Global score | Segmentation score | Steps | Images |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Regions | 0.534519 | [0.503827, 0.564630] | 0.458016 | 0.738527 | 17,187 | 549,984 |
| Consistency | 0.532381 | [0.503091, 0.562110] | 0.454424 | 0.740267 | 17,965 | 574,880 |
| Coverage | 0.526744 | [0.495695, 0.556632] | 0.448579 | 0.735184 | 13,627 | 436,064 |

Training wall times were 3,573.19, 3,572.93 and 3,573.22 seconds respectively, including final checkpoint overhead. Coverage processed about 24% fewer images than consistency within the same configured time limit. Region-context processed about 4.3% fewer images than consistency. Checkpoints, training summaries and task-level JSON files live under `/data/hassan/nanobrain/anatomical-regions-20261109/ryan-{regions,consistency,coverage}/`; final score artifacts are `eval/aggregate.json`.

Regions has the highest observed Ryan aggregate and global score; consistency has the highest observed segmentation score. Their task tradeoffs are:

- Regions has the highest point estimate on ABIDE diagnosis, ADHD diagnosis, CNP diagnosis, SOOP lesion Dice and UPenn one-year survival.
- Consistency has the highest point estimate on both age tasks, IDH status and both BraTS segmentation tasks. Its whole-tumor Dice is 0.865428 versus 0.857661 for regions; its IDH AUROC is 0.878336 versus 0.857437.
- Coverage has the highest point estimate only on SOOP poor-mRS outcome, by a very small margin: AUROC 0.739761 versus 0.739554 for regions. Improved regional sampling availability did not yield the highest overall benchmark score in this run.

The evaluator uses fixed paired bootstrap samples across models. Paired open-score differences are:

| Comparison | Point difference | 95% paired bootstrap interval |
| --- | ---: | --- |
| Regions − consistency | +0.002138 | [−0.005778, +0.009149] |
| Regions − coverage | +0.007775 | [−0.001225, +0.016397] |
| Consistency − coverage | +0.005637 | [−0.003473, +0.014997] |

Every interval includes zero. These are single-training-seed results; evaluation bootstrap intervals do not measure training-seed variation. The point ranking does not establish a reliable performance separation.

### Evaluation provenance detail

Installed Brainmarks metadata pins `8ef7115d519a1d1f13910c1e13bcae331b0ca3ad`; all 42 installed Python/YAML source/config files were additionally compared byte-for-byte with that pinned checkout and matched. The task JSON's generic `git` field resolves to the parent nanobrain repository because the installed package resides inside its `.venv`; it should not be read as the evaluator source revision. The task JSON's `cwd_git` field correctly records each frozen model worktree. The runner's `evaluation_commit` and installed `direct_url.json` record the actual evaluator pin.

The final trained-model named-region API check is handled by parent job 581890; smoke checkpoints already passed named embeddings and availability checks. No source, training recipe or model weights were changed during this final read-only monitoring/audit.
