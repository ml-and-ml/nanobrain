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
import subprocess

from finalize import collect


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "campaign/manifest.json").read_text())
OWNER = "ml-and-ml"
REMOTE = "https://github.com/ml-and-ml/nanobrain.git"
RECEIPTS = ROOT / "research/2026-11-09/labless-submissions.json"
spec = importlib.util.spec_from_file_location("upstream_submit", ROOT / "scripts/submit.py")
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)


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
        original_files = payload["repo"]["review_files"]["files"]
        source_hashes = {p: hashlib.sha256(c.encode()).hexdigest() for p, c in original_files.items()}
        original_files["pyproject.toml"] = subprocess.check_output(
            ["git", "show", f"{row['commit']}:pyproject.toml"], cwd=ROOT, text=True)
        original_files["submission/prepare_atlas.py"] = (ROOT / "campaign/atlas/prepare_atlas.py").read_text()
        original_files["submission/atlas-provenance.json"] = (ROOT / "campaign/atlas/provenance.json").read_text()
        original_files["submission/README.md"] = reproduction_notes(row)
        original_files["submission/source-hashes.json"] = json.dumps(source_hashes, indent=2) + "\n"
        payload["notes"] = (
            f"Independent {row['change']} experiment on {row['base']} base {row['base_commit']}. "
            f"Training seed 7338; one H100; original {'60' if row['base'] == 'main' else '59.5'}-minute schedule. "
            "All 11 fixed open tasks evaluated on dev, evaluator seed 0 and unchanged pinned Brainmarks. "
            "EXTERNAL ANATOMICAL PRIOR: all six variants use the public 21-channel Harvard-Oxford "
            "subcortical probabilistic atlas in MNI152NLin6Asym through dataset-provided classical "
            "ANTs mni_affine registration. No pretrained neural segmenter or new registration. "
            "Atlas eligibility is not claimed; submitted for maintainer review. "
            "Named-region specificity is not established by the open score. "
            "Exact training source and config are preserved; submission/ contains explicitly "
            "supplemental atlas preparation, provenance and reproduction instructions. "
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


def reproduction_notes(row):
    return f"""# Reproducing {row['name']}

Exact training commit: `{row['commit']}` on branch `experiments/{row['name']}`.
Public repository: {REMOTE}
Base commit: `{row['base_commit']}`.
Evaluator: `{MANIFEST['evaluation_commit']}`.

The `src/` files and `config.yaml` come unchanged from the training snapshot and
effective run configuration. Their hashes are in `submission/source-hashes.json`.
`pyproject.toml` is copied from the exact training commit; its `uv.lock` is in Git.
Files under `submission/` are supplemental reproduction material added after the
run, not changes to the trained model. No patient data or evaluation samples are
included. The original training summary retains the remote recorded at run time;
the exact experiment commit has since been published in the repository above.

Use the Brainmarks/FOMO data under its access terms and set BRAINMARKS_SMRI_DATA.
Check out the experiment branch, install its locked dependencies with `uv sync`,
and overlay the source/config supplied by Labless. The original committed image
filelist is retained by Git; do not regenerate or change it for this comparison.

Prepare the external atlas before fresh training:

```sh
uv run python submission/prepare_atlas.py --output ./atlas --download-dir ./atlas/source
export NANOBRAIN_ATLAS_PATH="$PWD/atlas/harvard_oxford_subcortical.npz"
uv run python -m nanobrain.train --config config.yaml output_dir=./output
```

Verify downloaded source hashes against `submission/atlas-provenance.json`.
Atlas source: https://www.nitrc.org/frs/download.php/9902/HarvardOxford.tgz
Atlas documentation and attribution: https://fsl.fmrib.ox.ac.uk/fsl/docs/other/datasets.html
This population atlas is external anatomical supervision, not a subject-specific
segmentation. Competition eligibility is left to maintainers; no validated status
is claimed. Existing dataset ANTs affine registrations are reused, without new
registration or a pretrained neural teacher. Model checkpoints contain the atlas
buffers, so inference from a checkpoint does not need this download.

Scores are from all 11 fixed open tasks on the dev split. Training seed is 7338;
evaluation seed is 0. Main and Ryan retain different original training recipes.
"""


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
