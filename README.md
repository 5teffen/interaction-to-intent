# From Interaction to Intent

Code to reproduce the results of *"From Interaction to Intent: Inferring User
Objectives from Provenance Logs"* ([arXiv:2607.04501](https://arxiv.org/abs/2607.04501)).

The data lives separately on Zenodo; this repository is **only** the pipeline that turns that
data into the paper's tables and figures. Two commands take you from a clean checkout to the
full set of results:

```bash
make data     # download the dataset from Zenodo into data/processed/
make all      # reproduce every table and figure
```

## What it does

A two-stage intent-classification pipeline over hover-interaction sequences:

1. **Stage 1 — atomic task classification.** Classify one hover sequence into one of seven
   analytical tasks (fine) or three interaction groups (coarse), with **XGBoost** on 39-dim
   summary features and a **BiGRU** on 24-dim per-timestep temporal features.
2. **Stage 2 — online multi-task segmentation.** Label the active task at every timestep of a
   composed multi-task session with a **UniGRU-CRF** sequence labeler.

## Setup

Requires Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Reproducing individual results

Each paper artifact has its own target. `make all` runs all of them.

| Command | Reproduces |
|---|---|
| `make table1` | **Table 1** — atomic (XGBoost/BiGRU) + online multi-task (UniGRU-CRF Variants A/B), 7- and 3-class, LOSO CV + held-out test |
| `make table2` | **Table 2** — XGBoost feature-group and BiGRU temporal ablations (incl. shuffled-order H4) |
| `make fig-confusion` | Per-model confusion matrices (run after `make table1`) |
| `make fig-dataset` | Dataset composition summary (task/dataset/projection counts, sequence lengths) |
| `make fig-composition` | Composed multi-task session statistics |
| `make figures` | All three figures |

Expected numbers to check your run against are in
[`results/REFERENCE.md`](results/REFERENCE.md).

### Parameters

Defaults reproduce the reported numbers (`SCHEME=fine`, `NORM=per-source`, `SEEDS=42`). They
combine freely, e.g. run only the 3-class ablations with three seeds:

```bash
make table2 SCHEME=coarse SEEDS=42,43,44
```

- `SCHEME` — `fine` (7 tasks) or `coarse` (3 interaction groups)
- `NORM` — `per-source` (reported) or `global` feature standardization
- `SEEDS` — comma-separated seeds for ablation / multi-seed runs

## Layout

```
src/                # library: features, models, training, metrics
scripts/            # CLI entry points invoked by the Makefile
results/REFERENCE.md# expected numbers for verifying reproduction
config.yaml         # canonical hyperparameters (documentation; Makefile passes them as flags)
Makefile            # pipeline orchestration
data/               # populated by `make data` from Zenodo; gitignored (no data is committed)
```

## Dataset

> **From Interaction to Intent: A Crowdsourced Provenance Dataset for Multidimensional Projection Exploration**
> DOI: [10.5281/zenodo.21207995](https://doi.org/10.5281/zenodo.21207995)

`make data` fetches and unpacks it automatically: the interaction sequences
(`train.csv` / `test.csv`) land in `data/processed/` as `train_filtered.csv` / `test_filtered.csv`,
and the projection coordinates land in `data/projected/`.

## License

Code: [MIT](LICENSE) · Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Citation

> **Status:** Paper under review; citation will be updated on acceptance.
> Preprint: https://arxiv.org/abs/2607.04501

```bibtex
@misc{holter2026interaction,
  title        = {From Interaction to Intent: Inferring User Objectives from Provenance Logs},
  author       = {Holter, Steffen and St{\"a}hle, Tobias and Narechania, Arpit and El-Assady, Mennatallah},
  year         = {2026},
  eprint       = {2607.04501},
  archivePrefix= {arXiv},
  note         = {Under review}
}

@dataset{holter2026dataset,
  title     = {From Interaction to Intent: A Crowdsourced Provenance Dataset for Multidimensional Projection Exploration},
  author    = {Holter, Steffen and St{\"a}hle, Tobias and Narechania, Arpit and El-Assady, Mennatallah},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21207995},
  url       = {https://doi.org/10.5281/zenodo.21207995}
}
```
