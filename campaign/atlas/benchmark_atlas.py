"""Measure atlas lookup cost without model training or patient images."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from atlas import AtlasPrior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    atlas = AtlasPrior().to(device)
    axes = [torch.arange(n, device=device) * 8 + 3.5 for n in (24, 30, 24)]
    centres = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(1, -1, 3)
    affine = torch.tensor([[1., 0, 0, -96], [0, 1., 0, -126], [0, 0, 1., -72], [0, 0, 0, 1.]], device=device)
    mni_affine = torch.eye(4, device=device)
    results = []
    for batch_size, patches in ((1, 17280), (32, 512), (32, 17280)):
        points = centres[:, :patches].expand(batch_size, -1, -1)
        timings = []
        for rep in range(args.repetitions + 1):
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            weights = atlas.patch_weights(points, affine, mni_affine)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if rep:
                timings.append(time.perf_counter() - started)
        results.append({"batch": batch_size, "patches": patches, "regions": 21,
                        "median_seconds": statistics.median(timings),
                        "output_bytes": weights.numel() * weights.element_size()})
    report = {"device": str(device), "cpu_threads": 1, "repetitions": args.repetitions,
              "atlas_lookup": "eight probability samples per patch; no registration", "measurements": results}
    output = Path(__file__).parent / ("benchmark_" + args.device.replace(":", "_") + ".json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
