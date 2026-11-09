"""Queue only committed experiments with successful matching GPU smoke checks."""

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "campaign/manifest.json").read_text())
parser = argparse.ArgumentParser()
parser.add_argument("names", nargs="*")
parser.add_argument("--attempt", default="1")
parser.add_argument("--eval-only", action="store_true")
args = parser.parse_args()
names = args.names or [r["name"] for r in manifest["experiments"]]
assert all(name in {r["name"] for r in manifest["experiments"]} for name in names)
for name in names:
    worktree = ROOT / "branches" / name
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=worktree, text=True).strip(), name
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
    smoke = json.loads((ROOT / "campaign/state" / f"smoke-{name}-{args.attempt}.json").read_text())
    assert smoke["status"] == "complete" and smoke["commit"] == commit, f"Smoke does not cover {name} at {commit}"
    state_path = ROOT / "campaign/state" / f"{name}.json"
    if state_path.exists() and not args.eval_only:
        raise RuntimeError(f"Refusing duplicate launch: {name}")
    log = ROOT / "campaign/logs" / f"{name}-%j.out"
    command = ["sbatch", "--parsable", f"--job-name=nb-{name}", f"--output={log}",
               str(ROOT / "campaign/run_experiment.sbatch"), name]
    if args.eval_only:
        command += ["--phase", "eval"]
    job = subprocess.check_output(command, text=True).strip()
    state = {"name": name, "status": "queued", "job_id": job, "commit": commit,
             "phase": "eval" if args.eval_only else "full", "submission": "disabled",
             "queued_at": dt.datetime.now(dt.UTC).isoformat()}
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(name, job, flush=True)
