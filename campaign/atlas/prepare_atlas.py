"""Download/publicly verify Harvard--Oxford and write the inference asset.

No patient images, metadata or benchmark labels are read by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np

URL = "https://www.nitrc.org/frs/download.php/9902/HarvardOxford.tgz"
ATLAS_MEMBER = "data/atlases/HarvardOxford/HarvardOxford-sub-prob-2mm.nii.gz"
LABEL_MEMBER = "data/atlases/HarvardOxford-Subcortical.xml"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/data/hassan/nanobrain/atlas"))
    parser.add_argument("--download-dir", type=Path, default=Path("/data/hassan/nanobrain/atlas/source"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.download_dir.mkdir(parents=True, exist_ok=True)
    archive = args.download_dir / "HarvardOxford.tgz"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as tar:
        for member_name in (ATLAS_MEMBER, LABEL_MEMBER):
            target = args.download_dir / member_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member_name).read())
    image_path = args.download_dir / ATLAS_MEMBER
    label_path = args.download_dir / LABEL_MEMBER
    image = nib.load(image_path)
    probabilities = np.asarray(image.dataobj, dtype=np.uint8)
    labels = sorted(ET.parse(label_path).findall(".//label"), key=lambda label: int(label.attrib["index"]))
    names = tuple(label.text.strip().lower().replace("ventrical", "ventricle").replace("-", "_").replace(" ", "_") for label in labels)
    assert probabilities.shape == (91, 109, 91, 21)
    assert probabilities.min() == 0 and probabilities.max() <= 100
    assert names[8] == "left_hippocampus" and names[18] == "right_hippocampus"
    assert nib.aff2axcodes(image.affine) == ("L", "A", "S")
    asset_path = args.output / "harvard_oxford_subcortical.npz"
    np.savez_compressed(asset_path, probabilities=probabilities, inverse_affine=np.linalg.inv(image.affine), region_names=np.asarray(names))
    report = {
        "atlas": "Harvard-Oxford Subcortical Structural Atlas", "source_url": URL,
        "atlas_space": "MNI152NLin6Asym", "resolution_mm": 2,
        "probabilities": "uint8 percentages, preserved from FSL/Nilearn source",
        "archive_sha256": sha256(archive), "nifti_sha256": sha256(image_path),
        "xml_sha256": sha256(label_path), "asset_sha256": sha256(asset_path),
        "shape": list(probabilities.shape), "affine": image.affine.tolist(),
        "region_names": names, "asset_bytes": asset_path.stat().st_size,
        "atlas_eligibility": "Not assumed Labless eligible; review external atlas prior rules before submitting.",
        "registration": "Reuse dataset-provided ANTs 12-DOF image-world-to-MNI152NLin6Asym affine. No new registration and no neural segmentation teacher.",
    }
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    (Path(__file__).parent / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
