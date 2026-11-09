"""Check actual FOMO input, checkpoint loading, dense and named feature outputs."""

import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from omegaconf import OmegaConf
import torch

from nanobrain.data import Fomo300kDataset, brle_to_dense
from nanobrain.model import vit_mae_3d


checkpoint = Path(sys.argv[1])
config = OmegaConf.load(checkpoint.parent / "config.yaml")
dataset = Fomo300kDataset("/data/brainmarks-smri/fomo300k", config.train_filelist)
sample = dataset[0]
mask = brle_to_dense(sample["mask_rle"], sample["shape"]).numpy()
image = np.zeros(sample["shape"], dtype=np.float32)
image[mask] = sample["values"].numpy().astype(np.float32) * sample["scale"]
image_input = {
    "image": nib.Nifti1Image(image, sample["affine"]),
    "brain_mask": nib.Nifti1Image(mask.astype(np.uint8), sample["affine"]),
    "mni_affine": np.asarray(sample["mni_affine"]),
}
model = vit_mae_3d(checkpoint=str(checkpoint)).cuda().eval()
prepared = model.transform(image_input)
result = model.forward_embeddings([prepared], return_dense=True)[0]
assert result["global_embedding"].shape == (768,)
assert torch.isfinite(result["global_embedding"]).all()
assert result["dense_embedding"].shape[-1] == 768
assert torch.isfinite(result["dense_embedding"]).all()
assert result["dense_mask"].any()
mode = config.model_kwargs.region_mode
report = {"mode": mode, "global_shape": list(result["global_embedding"].shape),
          "dense_shape": list(result["dense_embedding"].shape), "finite": True}
if mode == "regions":
    if hasattr(model, "forward_region_embeddings"):
        regions = model.forward_region_embeddings([prepared])[0]
        names, embeds, available = regions["names"], regions["embeddings"], regions["available"]
    else:
        names = result["region_names"]
        embeds = result["region_embeddings"]
        available = result["region_available"]
    assert "left_hippocampus" in names and "right_hippocampus" in names
    assert embeds.shape == (len(names), 768)
    assert torch.isfinite(embeds).all()
    assert available.any()
    assert torch.count_nonzero(embeds[~available]) == 0
    report["region_count"] = len(names)
    report["available_regions"] = int(available.sum())
path = checkpoint.parent / "checkpoint_check.json"
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
