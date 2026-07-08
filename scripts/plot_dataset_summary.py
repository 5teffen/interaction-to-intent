#!/usr/bin/env python3
"""
Dataset summary (Figure 1): the collected atomic provenance dataset.

Matches the original three-panel layout, now including the recipes dataset:
    (1) sequences per task
    (2) sequence length distribution (hovers)
    (3) dataset x projection (grouped bars)

Usage:
    python scripts/plot_dataset_summary.py \
        --csv data/processed/data-filtered.csv --out figures/dataset_summary
"""

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.constants import TASK_SHORT, N_CLASSES

BLUE, GREEN, RED = "#4C72B0", "#55A868", "#C44E52"
DATASETS = ["basketball", "pokemon", "weather", "recipes"]
PROJECTIONS = ["pca", "tsne", "umap"]
PROJ_COLORS = {"pca": BLUE, "tsne": GREEN, "umap": RED}


def main():
    p = argparse.ArgumentParser(description="Dataset summary figure")
    p.add_argument("--csv", default="data/processed/data-filtered.csv")
    p.add_argument("--out", default="figures/dataset_summary")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"  {len(df)} sequences, {df['prolific_id'].nunique()} participants")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2), constrained_layout=True)

    # (1) sequences per task
    tcounts = [int((df["task_type"] == i).sum()) for i in range(N_CLASSES)]
    ax[0].bar(range(N_CLASSES), tcounts, color=BLUE)
    ax[0].set_xticks(range(N_CLASSES))
    ax[0].set_xticklabels(TASK_SHORT, rotation=45, ha="right")
    ax[0].set_ylabel("Count")
    ax[0].set_title("Sequences per task")

    # (2) sequence length distribution
    ax[1].hist(df["sequence_length"].values, bins=28, color=BLUE)
    ax[1].set_xlabel("Sequence length (hovers)")
    ax[1].set_title("Sequence length distribution")

    # (3) dataset x projection grouped bars
    x = np.arange(len(DATASETS))
    width = 0.26
    for k, proj in enumerate(PROJECTIONS):
        vals = [int(((df["dataset"] == d) & (df["projection"] == proj)).sum())
                for d in DATASETS]
        ax[2].bar(x + (k - 1) * width, vals, width, label=proj, color=PROJ_COLORS[proj])
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(DATASETS, rotation=20, ha="right")
    ax[2].set_title("Dataset $\\times$ Projection")
    ax[2].legend(fontsize=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"  Saved {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
