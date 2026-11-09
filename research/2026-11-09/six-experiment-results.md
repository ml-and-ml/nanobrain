# Six anatomical-region experiment results

All six models completed full training and all 11 fixed open tasks on the dev split. **Nothing was submitted to Labless.**

The three changes are independent: learned contextual regions with patch feedback; same-region masked-view consistency; regional visible-patch quotas. Each was applied separately to main and Ryan's pinned model. Both final region-token checkpoints also passed a real-input named-region API check, returning 21 finite 768-dimensional vectors with explicit availability masks.

| Experiment | Open score | 95% bootstrap interval | Delta vs historical base | Train minutes | Steps |
|---|---:|---|---:|---:|---:|
| main-regions | 0.486434 | 0.457609–0.515164 | +0.013701 | 60.11 | 20,364 |
| main-consistency | 0.467114 | 0.437940–0.497189 | -0.005619 | 60.11 | 20,166 |
| main-coverage | 0.461273 | 0.432793–0.490943 | -0.011460 | 60.10 | 20,443 |
| ryan-regions | 0.534519 | 0.503827–0.564630 | -0.014954 | 59.55 | 17,187 |
| ryan-consistency | 0.532381 | 0.503091–0.562110 | -0.017092 | 59.55 | 17,965 |
| ryan-coverage | 0.526744 | 0.495695–0.556632 | -0.022729 | 59.55 | 13,627 |

Main's historical reference is 0.472733; Ryan's is 0.549473. These are published single-seed references, not freshly rerun controls. The intervals are the evaluator's sample-bootstrap intervals; they do not measure training-seed variation or establish a significant gain over the historical reference. Scores are normalized task metrics, not accuracy percentages.

## Interpretation

Learned contextual region tokens achieved the highest point score among the three new variants on both bases. Main + regions scored 0.486434, up 0.013701 against historical main, and led 9 of 11 tasks among the main variants. It is the only arm above its historical reference. Ryan + regions was the best of the six at 0.534519, but all three Ryan variants fell below Ryan's historical 0.549473. Paired sample-bootstrap comparisons among the Ryan variants all include zero; the point ranking is not evidence of a reliable difference.

Coverage improved regional availability in the smoke check but did not improve the final aggregate. Under Ryan's fixed time budget it processed 436,064 image presentations versus consistency's 574,880, about 24% fewer. Better anatomical coverage therefore came with a substantial throughput cost in that implementation. The next experiment should compare region tokens against a freshly rerun unchanged base at matched seeds and wall time, while separately testing the named vectors' anatomical specificity and downstream usefulness. These follow-up experiments have not been run.

## Per-task normalized scores

| Task | main-regions | main-consistency | main-coverage | ryan-regions | ryan-consistency | ryan-coverage |
|---|---:|---:|---:|---:|---:|---:|
| abide_diagnosis | 0.047758 | 0.031353 | 0.014851 | 0.130457 | 0.116391 | 0.079799 |
| adhd_diagnosis | -0.133486 | -0.150722 | -0.178002 | -0.054746 | -0.056730 | -0.089962 |
| cnp_diagnosis | 0.242507 | 0.186166 | 0.160542 | 0.284954 | 0.246960 | 0.281879 |
| ucsf_idh | 0.698391 | 0.671703 | 0.660911 | 0.714874 | 0.756672 | 0.740581 |
| upenn_survival_1y | 0.312628 | 0.275948 | 0.292726 | 0.263186 | 0.252744 | 0.254708 |
| ixi_age | 0.828211 | 0.813535 | 0.831960 | 0.912993 | 0.913774 | 0.911581 |
| openbhb_age | 0.813191 | 0.846486 | 0.829687 | 0.933304 | 0.935055 | 0.930524 |
| brats_enhancing_tumor | 0.617050 | 0.603514 | 0.598750 | 0.727859 | 0.732539 | 0.722219 |
| soop_lesion | 0.566313 | 0.525479 | 0.557092 | 0.630061 | 0.622836 | 0.627047 |
| soop_mrs_poor | 0.529593 | 0.507043 | 0.482244 | 0.479107 | 0.470526 | 0.479522 |
| brats_whole_tumor | 0.828624 | 0.827744 | 0.823240 | 0.857661 | 0.865428 | 0.856286 |

Classification scores are 2×AUROC−1, age regression uses R², and these segmentation tasks use mean per-subject Dice. The open score is the unweighted mean of these eleven values.

## Training and provenance

| Experiment | Image presentations | Parameters | Peak GPU GB | Source commit | Job |
|---|---:|---:|---:|---|---|
| main-regions | 651,648 | 104,153,345 | 34.54 | `46715080735c` | 581780 |
| main-consistency | 645,312 | 98,817,280 | 33.95 | `130d1aabf047` | 581781 |
| main-coverage | 654,176 | 98,817,280 | 33.94 | `695148c74e90` | 581782 |
| ryan-regions | 549,984 | 110,200,578 | 27.97 | `f65cb9583236` | 581784 |
| ryan-consistency | 574,880 | 104,864,513 | 27.51 | `770c81ab6acb` | 581785 |
| ryan-coverage | 436,064 | 104,864,513 | 27.50 | `2f3db395ab71` | 581787 |

Evaluator: `8ef7115d519a1d1f13910c1e13bcae331b0ca3ad`. All runs used training seed 7338, evaluator seed 0, and one H100. Main retained its 86,713-image list, AdamW and 60-minute schedule; Ryan retained its 253,418-image list, Muon, native-spacing training and 59.5-minute schedule. Different base recipes are not a controlled main-versus-Ryan architecture comparison. Image presentations count sampled volumes, not unique subjects. Raw training losses differ across objectives and should not be ranked across arms.

The atlas is external anatomical supervision: public Harvard–Oxford probability maps combined with the dataset-provided MNI affine. It is not a subject-specific segmentation, and its eligibility for a public competition submission has not been established. The open benchmark evaluates existing global and dense outputs; it does not directly validate the usefulness or anatomical specificity of the new named region vectors.

## Review artifacts

- [Protocol and branch folders](../../campaign/README.md)
- [Named contextual embedding API](../../campaign/REGION_EMBEDDINGS.md)
- [Atlas source and geometry checks](../../campaign/atlas/README.md)
- [Main validation and task comparison](../../campaign/main_audit.md)
- [Ryan validation and paired comparisons](../../campaign/ryan_audit.md)
- [Machine-readable scores and exact source/config provenance](six-experiment-results.json)
- Checkpoints and complete evaluation artifacts: `/data/hassan/nanobrain/anatomical-regions-20261109/<experiment>/`
