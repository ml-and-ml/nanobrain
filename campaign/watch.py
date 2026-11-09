"""Launch smoke-verified arms as they become ready and monitor aggregate progress."""

import datetime as dt
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "campaign/manifest.json").read_text())
deadline = time.monotonic() + 8 * 3600
while time.monotonic() < deadline:
    progress = []
    for spec in manifest["experiments"]:
        name = spec["name"]
        state_path = ROOT / "campaign/state" / f"{name}.json"
        smoke_path = ROOT / "campaign/state" / f"smoke-{name}-1.json"
        if not state_path.exists() and smoke_path.exists():
            smoke = json.loads(smoke_path.read_text())
            if smoke["status"] == "complete":
                subprocess.run(["python3", str(ROOT / "campaign/launch.py"), name], cwd=ROOT, check=True)
        state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "awaiting_smoke"}
        folder = Path(manifest["output_root"]) / name
        log = folder / "log.json"
        last = {}
        if log.exists():
            lines = log.read_text().splitlines()
            for line in reversed(lines[-3:]):
                try:
                    last = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        eval_path = folder / "eval"
        task_count = len([p for p in eval_path.glob("*.json") if p.name != "aggregate.json"])
        progress.append({"name": name, "status": state["status"], "step": last.get("step"),
                         "loss": round(last["loss"], 5) if "loss" in last else None,
                         "tasks": task_count, "open_score": state.get("open_score")})
    print(json.dumps({"at": dt.datetime.now(dt.UTC).isoformat(), "progress": progress}), flush=True)
    subprocess.run(["python3", str(ROOT / "campaign/report.py")], check=True, stdout=subprocess.DEVNULL)
    if all(p["status"] == "complete" for p in progress):
        break
    if any(p["status"] == "failed" for p in progress):
        print("An experiment failed; inspect its state and Slurm log. Other runs continue.", flush=True)
    time.sleep(30)
