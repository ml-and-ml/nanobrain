"""Collect only aggregate scores and run metadata; no patient-level data."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "campaign/manifest.json").read_text())
rows = []
for spec in manifest["experiments"]:
    path = ROOT / "campaign/state" / f"{spec['name']}.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    folder = Path(manifest["output_root"]) / spec["name"]
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    agg_path = folder / "eval/aggregate.json"
    aggregate = json.loads(agg_path.read_text()).get("summary", {}) if agg_path.exists() else {}
    score = aggregate.get("open", {}).get("score")
    row = {"experiment": spec["name"], "base": spec["base"], "change": spec["change"],
           "status": state.get("status", "not_started"), "open_score": score,
           "reference_score": spec["reference_score"],
           "delta_vs_reference": score - spec["reference_score"] if score is not None else None,
           "training_seconds": summary.get("wall_seconds"), "training_steps": summary.get("step"),
           "image_presentations": summary.get("images"), "commit": state.get("commit"),
           "job_id": state.get("job_id"), "run_dir": str(folder)}
    rows.append(row)
out = ROOT / "campaign/results"
out.mkdir(parents=True, exist_ok=True)
# Each run may finish concurrently. Replace full files atomically using process-unique temporaries.
import os
for suffix in ["json", "csv", "md"]:
    target = out / f"results.{suffix}"
    tmp = out / f".results.{os.getpid()}.{suffix}"
    if suffix == "json":
        tmp.write_text(json.dumps(rows, indent=2) + "\n")
    elif suffix == "csv":
        with tmp.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        lines = ["# Six-experiment results", "", "No Labless submissions. References are historical single-seed scores, not newly rerun controls.", "",
                 "| Experiment | Status | Open score | Delta vs reference | Train minutes | Job |",
                 "|---|---|---:|---:|---:|---|"]
        for r in rows:
            score = f"{r['open_score']:.6f}" if r['open_score'] is not None else "—"
            delta = f"{r['delta_vs_reference']:+.6f}" if r['delta_vs_reference'] is not None else "—"
            mins = f"{r['training_seconds']/60:.1f}" if r['training_seconds'] is not None else "—"
            lines.append(f"| {r['experiment']} | {r['status']} | {score} | {delta} | {mins} | {r['job_id'] or '—'} |")
        tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(target)
print(f"{sum(r['open_score'] is not None for r in rows)}/6 scores available: {out / 'results.md'}")
