# Six anatomical-region experiments

User instruction: implement each of three proposals separately on upstream main and Ryan's pinned `fbc1-s7338`; obtain six evaluations; **do not submit anything to Labless until user review**.

**Completed 2026-11-09:** all six full training runs and all 66 task evaluations finished. Both trained region-token checkpoints passed named/global/dense inference checks. See the [final scores, Labless links and provenance](../research/2026-11-09/six-experiment-results.md). Region tokens had the highest point score among the three variants on each base; only main + regions exceeded its historical reference. All six were subsequently submitted after user review; Labless accepted them as unvalidated. [Submission receipts](../research/2026-11-09/labless-submissions.json).

**Submission authorization:** after reviewing these results, the user instructed "Submit" and then "Submit all". All six exact experiment branches are published at [ml-and-ml/nanobrain](https://github.com/ml-and-ml/nanobrain). `submit_reviewed.py` prepares the six audited payloads, including explicit external-atlas disclosure and supplemental reproduction instructions. Run `.venv/bin/python campaign/submit_reviewed.py --submit` after Labless authentication; it records receipts and checks deterministic run IDs before retrying. This authorization applies to these six reviewed runs. The frozen training branches and their original submission guards remain unchanged.

All six public entries were independently read back: contributor, every metric, exact source commit and uploaded source/config contents match the local records. All are unvalidated, with empty validation errors. Labless's generated clone command currently names upstream MedARC despite correct fork metadata; for reproduction, clone `https://github.com/ml-and-ml/nanobrain.git` and check out the exact experiment branch/commit recorded in the receipt. The atlas setup and dependency links in each submission resolve to pinned public files.

## Isolated branches and directories

Each folder is a Git worktree with its own experiment branch and full training configuration:

| Folder under `branches/` | Base | Single added change |
|---|---|---|
| `main-regions` | upstream `440fc92` | Learned anatomical queries and regional/global context, fed back into patch features |
| `main-consistency` | upstream `440fc92` | Same-region consistency across the two existing masked views |
| `main-coverage` | upstream `440fc92` | Regional quotas within the original visible-patch budget |
| `ryan-regions` | Ryan `60df476` | Learned anatomical queries and regional/global context, fed back into patch features |
| `ryan-consistency` | Ryan `60df476` | Same-region consistency across the two existing masked views |
| `ryan-coverage` | Ryan `60df476` | Regional quotas within the original visible-patch budget |

These are independent arms, not cumulative changes. The consistency arm uses anatomical weighted pooling without adding the learned query head. Coverage changes sampling without adding a new model head or objective. All three share only the anatomy lookup infrastructure.

## Anatomy and data

Brainmarks already supplies a classical 12-parameter ANTs `mni_affine`, mapping image-world RAS millimeters to MNI152NLin6Asym, for both pretraining and evaluation inputs. Its model contract explicitly exposes this field. The original nanobrain loaders discarded it; the variants preserve it through preprocessing.

The Harvard–Oxford subcortical probability atlas supplies 21 population regions, including separate left/right hippocampus, amygdala, thalamus and ventricles. Atlas membership is approximate anatomical prior information, not a subject-specific segmentation or benchmark target. We use the dataset-provided registrations, with no additional pretrained neural segmenter or registration computation. Asset provenance and geometry checks are under `atlas/`.

The initial analysis did not inspect this available registration field; this discovery removes the proposed new-registration preprocessing requirement. Atlas eligibility for an eventual public competition submission remains a review item. None is being submitted.

Each base retains its own full file list, architecture, optimizer, seed 7338 and training schedule: main selects 86,713 images, while Ryan's QC file selects 253,418. Main runs for up to 60 minutes and Ryan for 59.5 minutes, matching the pinned recipes. A single H100 is allocated per experiment. These differing base recipes should not be interpreted as a controlled main-versus-Ryan architecture comparison.

## Execution and review

1. CPU behavior/geometry tests check anatomical support, missing regions, feedback gradients, anti-collapse behavior, disjoint sampling, and native-spacing transforms.
2. `smoke_all.sbatch` runs each full-width model for 20 steps on actual FOMO data and checks a saved checkpoint's global/dense output and named regions where applicable.
3. `launch.py` accepts only clean, committed worktrees whose exact commits passed smoke checks, and queues the six one-GPU training/evaluation jobs.
4. `run.py` evaluates the last checkpoint on all 11 fixed `open` tasks using pinned Brainmarks `8ef7115`. It does not modify the evaluator or enable the private test subset.
5. `report.py` writes `results/results.md`, `.csv`, and `.json`. Full checkpoints, logs and individual task files live under `/data/hassan/nanobrain/anatomical-regions-20261109/`.
6. `finalize.py` verifies all six complete runs, exact source/config provenance, the fixed task set and both final named-region checks, then freezes aggregate-only Markdown/CSV/JSON reports under `research/2026-11-09/`.

The shared environment is `.venv/`; `environment.txt` records its packages. Each process uses its worktree's `src/` via `PYTHONPATH`. Its source snapshot and committed config identify the code actually trained.

Labless submission is absent from the training campaign launcher, removed from each branch's standard training launcher, and blocked at the original direct submission CLI. After user review, the separate `submit_reviewed.py` helper authenticated and submitted the six completed runs. Labless allows only specific source paths: exact source/config snapshots are uploaded, while supplemental atlas setup and dependencies are linked to pinned public Git commits.

The reference scores in the results table are historical single-seed results (main 0.472733; Ryan 0.549473), not freshly rerun controls. Report failures and per-task tradeoffs honestly; do not relabel a smoke test or incomplete evaluation as one of the six requested results.
