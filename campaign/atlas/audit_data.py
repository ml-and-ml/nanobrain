"""Read a deterministic FOMO subset and report only aggregate anatomy QA.

No individual image, filename, subject identifier or raw data is written.
"""

import argparse
import io
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--root", type=Path, default=Path("/data/brainmarks-smri/fomo300k"))
    parser.add_argument("--filelist", type=Path, default=Path("/admin/home/hm/repos/nanobrain/src/nanobrain/config/fomo300k_images.txt"))
    args = parser.parse_args()
    atlas_dir = Path(__file__).parent
    asset = np.load("/data/hassan/nanobrain/atlas/harvard_oxford_subcortical.npz")
    probabilities = asset["probabilities"].astype(np.float32)
    atlas_affine = np.linalg.inv(asset["inverse_affine"])
    names = asset["region_names"].tolist()
    available = np.any(probabilities > 0, axis=-1)
    indices = np.argwhere(available)
    world = nib.affines.apply_affine(atlas_affine, indices)
    probabilities = probabilities[available]
    denominators = probabilities.sum(axis=0)
    members = pd.read_parquet(args.root / "tables/shards.parquet", columns=["path", "shard", "offset", "size"]).set_index("path")
    paths = args.filelist.read_text().split()
    rows = members.loc[[paths[i] for i in np.linspace(0, len(paths) - 1, args.count, dtype=int)]]
    fractions = []
    lat_errors = 0
    elapsed = []
    for _, row in rows.iterrows():
        started = time.perf_counter()
        with open(args.root / row["shard"], "rb") as shard:
            shard.seek(row["offset"])
            data = shard.read(row["size"])
        with np.load(io.BytesIO(data)) as sample:
            mask = np.repeat(np.arange(len(sample["mask_rle"])) % 2 == 1, sample["mask_rle"]).reshape(sample["shape"])
            transform = np.linalg.inv(sample["mni_affine"] @ sample["affine"])
            voxels = nib.affines.apply_affine(transform, world)
            inside = map_coordinates(mask.astype(np.uint8), voxels.T, order=0, mode="constant", cval=0)
            fractions.append((probabilities * inside[:, None]).sum(axis=0) / denominators)
            # Full inverse composition must map template coordinates back exactly.
            recovered = nib.affines.apply_affine(sample["mni_affine"] @ sample["affine"], voxels)
            lat_errors += int(np.max(np.abs(recovered - world)) > 1e-5)
        elapsed.append(time.perf_counter() - started)
    fractions = np.asarray(fractions)
    report = {
        "data": "FOMO300K pretraining filelist only; deterministic evenly spaced subset",
        "samples_checked": len(fractions), "geometry_roundtrip_failures": lat_errors,
        "template_atlas_probability_inside_subject_brain_mask": {
            name: {"median": float(np.median(fractions[:, r])), "minimum": float(fractions[:, r].min())}
            for r, name in enumerate(names)
        },
        "all_region_mass_fraction_median": float(np.median(fractions)),
        "sample_read_and_cpu_geometry_seconds_median": float(np.median(elapsed)),
        "interpretation": "Brain-mask containment and geometry checks only. This does not validate hippocampus segmentation accuracy or individual anatomy.",
    }
    (atlas_dir / "data_qa.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
