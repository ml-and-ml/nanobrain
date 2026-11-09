"""Run a pinned experiment locally on Slurm. This file never submits to Labless."""

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "campaign/manifest.json").read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=[r["name"] for r in MANIFEST["experiments"]])
    parser.add_argument("--phase", choices=["full", "smoke", "eval"], default="full")
    parser.add_argument("--attempt", default="1")
    args = parser.parse_args()
    spec = next(r for r in MANIFEST["experiments"] if r["name"] == args.name)
    worktree = ROOT / "branches" / args.name
    config = worktree / "experiment.yaml"
    if not config.is_file():
        raise RuntimeError(f"Missing committed experiment config: {config}")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=worktree, text=True)
    if dirty.strip():
        raise RuntimeError(f"Refusing to train an uncommitted worktree: {args.name}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
    url = importlib.metadata.distribution("brainmarks-smri").read_text("direct_url.json")
    eval_commit = json.loads(url)["vcs_info"]["commit_id"]
    assert eval_commit == MANIFEST["evaluation_commit"], (eval_commit, MANIFEST["evaluation_commit"])
    run_name = args.name if args.phase != "smoke" else f"smoke-{args.name}-{args.attempt}"
    run_dir = Path(MANIFEST["output_root"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(worktree / "src")
    env["BRAINMARKS_SMRI_DATA"] = MANIFEST["data_root"]
    env["NANOBRAIN_LABLESS_SUBMISSION"] = "disabled"
    env["WANDB_MODE"] = "disabled"
    env.pop("BRAINMARKS_SMRI_TEST_SUBSET", None)
    state = {
        "name": args.name, "phase": args.phase, "run_name": run_name,
        "commit": commit, "base_commit": spec["base_commit"],
        "evaluation_commit": eval_commit,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "run_dir": str(run_dir), "job_id": env.get("SLURM_JOB_ID"),
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
        "submission": "disabled", "status": "starting",
    }
    state_path = ROOT / "campaign/state" / f"{run_name}.json"
    current = None

    def on_term(signum, frame):
        state["status"] = "interrupted"
        save(state_path, state)
        if current is not None:
            current.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, on_term)
    save(state_path, state)

    def run(command, phase):
        nonlocal current
        state["status"] = phase
        state["command"] = command
        save(state_path, state)
        print(json.dumps({"phase": phase, "command": command}), flush=True)
        current = subprocess.Popen(command, cwd=worktree, env=env)
        code = current.wait()
        current = None
        if code:
            state["status"] = "failed"
            state["failed_phase"] = phase
            state["returncode"] = code
            save(state_path, state)
            raise SystemExit(code)
        if state["status"] == "interrupted":
            raise SystemExit(143)

    python = sys.executable
    if args.phase != "eval":
        command = [python, "-u", "-m", "nanobrain.train", "--config", str(config), "--overrides",
                   f"name={run_name}", f"output_dir={MANIFEST['output_root']}", "wandb=false"]
        if args.phase == "smoke":
            command += ["total_steps=20", "warmup_steps=5", "time_limit=10", "compile=false",
                        "plot_period=0", "checkpoint_period=1000000", "print_freq=1"]
            if spec["base"] == "ryan":
                command += ["epochs=null", "warmup_epochs=null", "decay_epochs=null", "time_schedule=false"]
        run(command, "training")
        summary = json.loads((run_dir / "summary.json").read_text())
        if not summary.get("done"):
            state["status"] = "interrupted"
            save(state_path, state)
            raise SystemExit(143)

    if args.phase == "smoke":
        run([python, str(ROOT / "campaign/check_checkpoint.py"), str(run_dir / "checkpoint-last.pth")],
            "checking_checkpoint")
    else:
        run([python, "-u", "-m", "brainmarks_smri", "vit_mae_3d", "open", "--overrides",
             "imports=[nanobrain.model]", f"model_kwargs.checkpoint={run_dir / 'checkpoint-last.pth'}",
             f"output_dir={run_dir}", "name=eval"], "evaluating")
        aggregate = json.loads((run_dir / "eval/aggregate.json").read_text())
        if "open" not in aggregate["summary"]:
            raise RuntimeError("Incomplete evaluation: no 11-task open aggregate")
        state["open_score"] = aggregate["summary"]["open"]["score"]
    state["status"] = "complete"
    state["finished_at"] = dt.datetime.now(dt.UTC).isoformat()
    save(state_path, state)
    print(json.dumps(state, indent=2), flush=True)
    subprocess.run([python, str(ROOT / "campaign/report.py")], check=True)


if __name__ == "__main__":
    main()
