"""Freeze the six complete aggregate results for review, without patient-level data."""

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "campaign/manifest.json").read_text())
# Exact open task set from the pinned evaluator in MANIFEST.
TASKS = [
    "abide_diagnosis", "adhd_diagnosis", "cnp_diagnosis", "ucsf_idh",
    "upenn_survival_1y", "ixi_age", "openbhb_age", "brats_enhancing_tumor",
    "soop_lesion", "soop_mrs_poor", "brats_whole_tumor",
]


def collect():
    rows = []
    for spec in MANIFEST["experiments"]:
        name = spec["name"]
        state = json.loads((ROOT / f"campaign/state/{name}.json").read_text())
        if state.get("status") != "complete":
            raise RuntimeError(f"{name} is {state.get('status')}; six full results are required")
        folder = Path(state["run_dir"])
        summary = json.loads((folder / "summary.json").read_text())
        aggregate = json.loads((folder / "eval/aggregate.json").read_text())["summary"]
        if not summary["done"] or summary["stopped_by_signal"]:
            raise RuntimeError(f"{name}: training did not complete")
        if state["commit"] != summary["git"]["sha"] or summary["git"]["dirty"]:
            raise RuntimeError(f"{name}: training provenance mismatch")
        if state["evaluation_commit"] != MANIFEST["evaluation_commit"]:
            raise RuntimeError(f"{name}: evaluator revision mismatch")
        config = ROOT / "branches" / name / "experiment.yaml"
        if hashlib.sha256(config.read_bytes()).hexdigest() != state["config_sha256"]:
            raise RuntimeError(f"{name}: experiment config changed")
        if set(TASKS) - aggregate.keys():
            raise RuntimeError(f"{name}: incomplete open evaluation")
        score = aggregate["open"]["score"]
        if not math.isfinite(score) or not math.isclose(score, state["open_score"]):
            raise RuntimeError(f"{name}: inconsistent or nonfinite aggregate")
        if not math.isclose(score, sum(aggregate[t]["score"] for t in TASKS) / len(TASKS)):
            raise RuntimeError(f"{name}: aggregate is not the fixed 11-task mean")
        if any(not math.isfinite(aggregate[t]["score"]) for t in TASKS):
            raise RuntimeError(f"{name}: nonfinite task score")
        evaluation_seconds = 0.0
        for task in TASKS:
            result = json.loads((folder / "eval" / f"{task}.json").read_text())
            if result["subset"] != "dev" or result["task"] != task:
                raise RuntimeError(f"{name}/{task}: wrong evaluation task or subset")
            if result["config"]["seed"] != 0:
                raise RuntimeError(f"{name}/{task}: evaluator seed changed")
            # The pinned evaluator stores git --short, while training records the full SHA.
            evaluation_sha = result["cwd_git"]["sha"]
            if len(evaluation_sha) < 7 or not state["commit"].startswith(evaluation_sha) or result["cwd_git"]["dirty"]:
                raise RuntimeError(f"{name}/{task}: evaluation source mismatch")
            if result["config"]["model_kwargs"]["checkpoint"] != str(folder / "checkpoint-last.pth"):
                raise RuntimeError(f"{name}/{task}: evaluation used a different checkpoint")
            evaluation_seconds += result["seconds"]
        named_regions_checked = None
        if spec["change"] == "regions":
            check = json.loads((folder / "checkpoint_check.json").read_text())
            if not check["finite"] or check["mode"] != "regions" or check["region_count"] != 21:
                raise RuntimeError(f"{name}: final named-region output check failed")
            named_regions_checked = check["region_count"]
        rows.append({
            **spec, "commit": state["commit"], "job_id": state["job_id"],
            "config_sha256": state["config_sha256"], "run_dir": str(folder),
            "checkpoint": str(folder / "checkpoint-last.pth"),
            "open_score": score, "open_low": aggregate["open"]["low"],
            "open_high": aggregate["open"]["high"],
            "delta_vs_historical_reference": score - spec["reference_score"],
            "training_seconds": summary["wall_seconds"], "training_steps": summary["step"],
            "image_presentations": summary["images"], "num_params": summary["num_params"],
            "peak_memory_gb": summary["peak_memory_gb"], "gpu": summary["gpu"],
            "evaluation_seconds": evaluation_seconds,
            "named_regions_checked": named_regions_checked,
            "tasks": {t: aggregate[t] for t in TASKS},
        })
    return rows


def main():
    rows = collect()
    out = ROOT / "research/2026-11-09"
    payload = {"finished_at": dt.datetime.now(dt.UTC).isoformat(),
               "evaluation_commit": MANIFEST["evaluation_commit"],
               "subset": "dev", "training_seed": MANIFEST["seed"], "evaluation_seed": 0,
               "submission": "disabled",
               "results": rows}
    (out / "six-experiment-results.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (out / "six-experiment-results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "tasks"])
        writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if k != "tasks"} for r in rows)
    with (out / "six-experiment-task-scores.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["task", *[r["name"] for r in rows]])
        writer.writerows([t, *[r["tasks"][t]["score"] for r in rows]] for t in TASKS)
    lines = ["# Six anatomical-region experiment results", "",
             "All six models completed full training and all 11 fixed open tasks on the dev split. **Nothing was submitted to Labless.**", "",
             "The three changes are independent: learned contextual regions with patch feedback; same-region masked-view consistency; regional visible-patch quotas. Each was applied separately to main and Ryan's pinned model. Both final region-token checkpoints also passed a real-input named-region API check, returning 21 finite 768-dimensional vectors with explicit availability masks.", "",
             "| Experiment | Open score | 95% bootstrap interval | Delta vs historical base | Train minutes | Steps |",
             "|---|---:|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['open_score']:.6f} | {r['open_low']:.6f}–{r['open_high']:.6f} | {r['delta_vs_historical_reference']:+.6f} | {r['training_seconds']/60:.2f} | {r['training_steps']:,} |")
    lines += ["", "Main's historical reference is 0.472733; Ryan's is 0.549473. These are published single-seed references, not freshly rerun controls. The intervals are the evaluator's sample-bootstrap intervals; they do not measure training-seed variation or establish a significant gain over the historical reference. Scores are normalized task metrics, not accuracy percentages.", "",
              "## Interpretation", "",
              "Learned contextual region tokens achieved the highest point score among the three new variants on both bases. Main + regions scored 0.486434, up 0.013701 against historical main, and led 9 of 11 tasks among the main variants. It is the only arm above its historical reference. Ryan + regions was the best of the six at 0.534519, but all three Ryan variants fell below Ryan's historical 0.549473. Paired sample-bootstrap comparisons among the Ryan variants all include zero; the point ranking is not evidence of a reliable difference.", "",
              "Coverage improved regional availability in the smoke check but did not improve the final aggregate. Under Ryan's fixed time budget it processed 436,064 image presentations versus consistency's 574,880, about 24% fewer. Better anatomical coverage therefore came with a substantial throughput cost in that implementation. The next experiment should compare region tokens against a freshly rerun unchanged base at matched seeds and wall time, while separately testing the named vectors' anatomical specificity and downstream usefulness. These follow-up experiments have not been run.", "",
              "## Per-task normalized scores", "",
              "| Task | " + " | ".join(r["name"] for r in rows) + " |",
              "|---|" + "---:|" * len(rows)]
    for t in TASKS:
        lines.append("| " + t + " | " + " | ".join(f"{r['tasks'][t]['score']:.6f}" for r in rows) + " |")
    lines += ["", "Classification scores are 2×AUROC−1, age regression uses R², and these segmentation tasks use mean per-subject Dice. The open score is the unweighted mean of these eleven values.", "",
              "## Training and provenance", "",
              "| Experiment | Image presentations | Parameters | Peak GPU GB | Source commit | Job |",
              "|---|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['image_presentations']:,} | {r['num_params']:,} | {r['peak_memory_gb']:.2f} | `{r['commit'][:12]}` | {r['job_id']} |")
    lines += ["", f"Evaluator: `{MANIFEST['evaluation_commit']}`. All runs used training seed {MANIFEST['seed']}, evaluator seed 0, and one H100. Main retained its 86,713-image list, AdamW and 60-minute schedule; Ryan retained its 253,418-image list, Muon, native-spacing training and 59.5-minute schedule. Different base recipes are not a controlled main-versus-Ryan architecture comparison. Image presentations count sampled volumes, not unique subjects. Raw training losses differ across objectives and should not be ranked across arms.", "",
              "The atlas is external anatomical supervision: public Harvard–Oxford probability maps combined with the dataset-provided MNI affine. It is not a subject-specific segmentation, and its eligibility for a public competition submission has not been established. The open benchmark evaluates existing global and dense outputs; it does not directly validate the usefulness or anatomical specificity of the new named region vectors.", "",
              "## Review artifacts", "",
              "- [Protocol and branch folders](../../campaign/README.md)",
              "- [Named contextual embedding API](../../campaign/REGION_EMBEDDINGS.md)",
              "- [Atlas source and geometry checks](../../campaign/atlas/README.md)",
              "- [Main validation and task comparison](../../campaign/main_audit.md)",
              "- [Ryan validation and paired comparisons](../../campaign/ryan_audit.md)",
              "- [Machine-readable scores and exact source/config provenance](six-experiment-results.json)",
              f"- Checkpoints and complete evaluation artifacts: `{MANIFEST['output_root']}/<experiment>/`", ""]
    (out / "six-experiment-results.md").write_text("\n".join(lines))
    subprocess.run(["python3", str(ROOT / "campaign/report.py")], check=True)
    print(out / "six-experiment-results.md")


if __name__ == "__main__":
    main()
