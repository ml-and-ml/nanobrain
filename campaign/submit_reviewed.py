"""Prepare or submit exactly the six runs approved by the user after review.

Keeps trained worktrees and checkpoints unchanged. Reuses the upstream payload
format, with explicit atlas disclosure and supplemental reproduction material.
"""

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

from finalize import collect


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "campaign/manifest.json").read_text())
OWNER = "ml-and-ml"
REMOTE = "https://github.com/ml-and-ml/nanobrain.git"
RECEIPTS = ROOT / "research/2026-11-09/labless-submissions.json"
spec = importlib.util.spec_from_file_location("upstream_submit", ROOT / "scripts/submit.py")
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
REPRODUCTION_COMMIT = "b96bafb"


def prepare(rows):
    installed = json.loads(importlib.metadata.distribution("brainmarks-smri").read_text("direct_url.json"))
    evaluator = installed["vcs_info"]["commit_id"]
    if evaluator != MANIFEST["evaluation_commit"]:
        raise RuntimeError("Installed evaluator does not match the six reviewed runs")
    prepared = []
    for row in rows:
        folder = Path(row["run_dir"])
        payload = upstream.make_payload(folder, evaluator)
        if payload["run"]["validation_errors"]:
            raise RuntimeError(payload["run"]["validation_errors"])
        if payload["run"]["metrics"]["open"] != row["open_score"]:
            raise RuntimeError(f"{row['name']}: score mismatch")
        # This is the published location of the exact training commit. The original
        # summary.git remains untouched and records the remote at training time.
        payload["repo"]["remote"] = REMOTE
        payload["contributor"]["login"] = OWNER
        # Labless restricts review file paths. Publish supplemental material in
        # Git and link it from notes; only the unchanged snapshot/config is uploaded.
        payload["notes"] = (
            f"Independent {row['change']} experiment on {row['base']} base {row['base_commit']}. "
            f"Training seed 7338; one H100; original {'60' if row['base'] == 'main' else '59.5'}-minute schedule. "
            "All 11 fixed open tasks evaluated on dev, evaluator seed 0 and unchanged pinned Brainmarks. "
            "EXTERNAL ANATOMICAL PRIOR: all six variants use the public 21-channel Harvard-Oxford "
            "subcortical probabilistic atlas in MNI152NLin6Asym through dataset-provided classical "
            "ANTs mni_affine registration. No pretrained neural segmenter or new registration. "
            "Atlas eligibility is not claimed; submitted for maintainer review. "
            "Named-region specificity is not established by the open score. "
            "Exact training source and config are preserved. Supplemental atlas setup: "
            f"https://github.com/{OWNER}/nanobrain/blob/{REPRODUCTION_COMMIT}/campaign/atlas/README.md . "
            f"Preparation script: https://github.com/{OWNER}/nanobrain/blob/{REPRODUCTION_COMMIT}/campaign/atlas/prepare_atlas.py . "
            f"Dependencies and lockfile: https://github.com/{OWNER}/nanobrain/tree/{row['commit']} . "
            "Before fresh training, run the linked prepare_atlas.py and set NANOBRAIN_ATLAS_PATH "
            "to the resulting harvard_oxford_subcortical.npz. The saved checkpoint embeds the atlas. "
            "Single-seed ablation; historical reference scores are not fresh matched controls. "
            "User reviewed the six scores and authorized all six submissions."
        )
        payload["run"]["ended_at"] = dt.datetime.fromtimestamp(
            (folder / "summary.json").stat().st_mtime, dt.UTC).isoformat()
        payload.pop("submission_id", None)
        payload["submission_id"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
        path = folder / "labless_submission.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        prepared.append((row, payload))
        print(f"Prepared {row['name']}: score={row['open_score']:.9f}, id={payload['submission_id']}", flush=True)
    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Upload the six user-reviewed runs")
    args = parser.parse_args()
    rows = collect()
    payloads = prepare(rows)
    if not args.submit:
        print("All six payloads prepared; no submission attempted.")
        return
    token_file = Path.home() / ".cache/nanobrain/labless_token.json"
    token = json.loads(token_file.read_text())["access_token"]
    login = upstream.api_request("GET", "/api/auth/github/me", token=token)["login"]
    if login != OWNER:
        raise RuntimeError(f"Expected {OWNER}, authenticated as {login}")
    receipts = json.loads(RECEIPTS.read_text()) if RECEIPTS.exists() else {
        "authorization": "User: Submit; clarified: Submit all, after reviewing six completed results.",
        "repository": REMOTE,
        "runs": [],
    }
    existing = {r["submission_id"]: r for r in receipts["runs"]}
    for row, payload in payloads:
        submission_id = payload["submission_id"]
        if submission_id in existing:
            print(f"Already submitted {row['name']}: {existing[submission_id]['url']}", flush=True)
            continue
        # Query the deterministic ID before retrying a possibly interrupted upload.
        run_id = f"run_sub_{submission_id}"
        try:
            upstream.api_request("GET", f"/api/runs/{run_id}")
            result = {"run_id": run_id, "recovered_existing": True}
        except RuntimeError as exc:
            if "failed: 404 " not in str(exc):
                raise
            result = upstream.api_request("POST", f"/api/nano-projects/{upstream.PROJECT}/submissions", payload, token)
        receipt = {
            "experiment": row["name"], "submission_id": submission_id,
            "run_id": run_id, "url": f"https://labless.dev/runs/{run_id}",
            "score": row["open_score"], "commit": row["commit"],
            "submitted_at": dt.datetime.now(dt.UTC).isoformat(), "response": result,
        }
        receipts["runs"].append(receipt)
        RECEIPTS.write_text(json.dumps(receipts, indent=2) + "\n")
        print(f"Submitted {row['name']}: {receipt['url']}", flush=True)


if __name__ == "__main__":
    main()
