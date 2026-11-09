# nanobrain

**Six experiments completed:** [final results and interpretation](research/2026-11-09/six-experiment-results.md), [branch layout and protocol](campaign/README.md), and [named region embeddings](campaign/REGION_EMBEDDINGS.md). Each of three changes was trained and evaluated independently on main and Ryan's pinned model in six folders under `branches/`. Labless submission is disabled pending user review.

A collaborative (or competitive) challenge to find the best recipe for training brain MRI foundation models. Inspired by [nanoGPT](https://github.com/karpathy/nanogpt), [modded-nanogpt](https://github.com/kellerjordan/modded-nanogpt) and our sibling project [nanopath](https://github.com/MedARC-AI/nanopath).

**Want to get involved? Join us in the [MedARC Discord](https://discord.gg/tVR4TWnRM9) (find us in #neuro-fm)!**

## Setup

```sh
git clone https://github.com/MedARC-AI/nanobrain && cd nanobrain
uv sync
```

Training and eval both read the data folder in `BRAINMARKS_SMRI_DATA`. On the MedARC cluster:

```sh
echo "BRAINMARKS_SMRI_DATA=/data/brainmarks-smri" >> .env
```

Elsewhere, download it from the gated Hugging Face repo [medarc/brainmarks-smri](https://huggingface.co/datasets/medarc/brainmarks-smri) (~850 GB). Accept its terms first, then:

```sh
uv run hf auth login
uv run hf download medarc/brainmarks-smri --repo-type dataset --local-dir data
echo "BRAINMARKS_SMRI_DATA=data" >> .env
```

## Train and evaluate locally

```sh
python3 campaign/report.py
# For the six fixed variants, use the launch protocol in campaign/README.md.
```

The six variant folders are separate Git worktrees with their own committed `experiment.yaml`. `campaign/run.py` trains each recipe, then evaluates its last checkpoint on all brainmarks-smri `open` tasks. Checkpoints and full evaluation artifacts stay under `/data/hassan/nanobrain/anatomical-regions-20261109/`.

Labless submission is disabled in this checkout and all six experiment branches until the user reviews the results. The [upstream challenge](https://labless.dev/nano-projects/nanobrain-smri) remains the source of its competition rules; atlas eligibility has not been established.

## Rules

- Training gets 1 hour of wall time on one H100 (~3.6e18 theoretical FLOPs).
- Pretrain only on the FOMO300K images in brainmarks-smri.
- The score is the mean over the fixed 11 brainmarks-smri `open` tasks.
- No weights or teachers pretrained on brain MRI.
- Test-time tricks on the embeddings (combining layers, pooling strategies, test-time augmentation) are technically allowed but lose style points.

Anything else can change: data curation, model, objective, preprocessing, schedule, etc.

## Baseline

The baseline is a 3D ViT-B masked autoencoder trained on 87K MRI brain volumes (T1w, T2w, FLAIR, DWI) at 1mm isotropic resolution with an 8mm patch size and an effective masking ratio of ~80%.

## Code

| | |
|---|---|
| [`train.py`](src/nanobrain/train.py) | training loop |
| [`model.py`](src/nanobrain/model.py) | ViT MAE and the brainmarks-smri model contract |
| [`data.py`](src/nanobrain/data.py) | FOMO300K dataset and preprocessing |
| [`default_pretrain.yaml`](src/nanobrain/config/default_pretrain.yaml) | default config |
| [`select_images.py`](src/nanobrain/select_images.py) | pretraining image selection, writes `fomo300k_images.txt` |
| [`train.sbatch`](scripts/train.sbatch) | train, then evaluate |
| [`submit.py`](scripts/submit.py) | leaderboard submission blocked pending user review |
