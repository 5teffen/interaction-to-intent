#!/usr/bin/env python3
"""
Render row-normalized confusion matrices (7-class) as a publication figure.

Aggregates the per-fold confusion matrices stored in the LOSO CV checkpoints and
plots three panels:
    (a) XGBoost + Summary context        (atomic)
    (b) BiGRU + Temporal context         (atomic)
    (c) UniGRU+CRF + Log stitching       (online multi-task, per-timestep)

Usage (defaults pick the latest matching checkpoint; override for a specific run):
    python scripts/plot_confusion.py \
        --xgb     checkpoints/xgboost_fine_loso_cv_YYYYMMDD.pkl \
        --bigru   checkpoints/bigru_fine_loso_cv_YYYYMMDD.pkl \
        --unigru  checkpoints/stage2_A_fine_loso_cv_YYYYMMDD.pkl \
        --out figures/confusion_matrices
"""

import argparse
import glob
import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.constants import TASK_SHORT

# Display order grouping tasks by behavioral category (original task indices):
#   A Local Exploration : Name Cluster(1), Find Similar(2), Identify Outliers(3)
#   B Global Scanning   : Generate Clusters(0), Add Data(4)
#   C Comparative Probing: Map Dimensions(6), Compare Clusters(5)
ORDER = [1, 2, 3, 0, 4, 6, 5]
GROUP_SEPS = [2.5, 4.5]  # between A|B and B|C in the reordered layout


def _latest(pattern):
    matches = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime)
    return matches[-1] if matches else None


def aggregated_cm(ckpt_path):
    """Sum the per-fold confusion matrices in a CV checkpoint."""
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    fold_results = ckpt["fold_results"]
    if "confusion_matrix" not in fold_results[0]:
        raise KeyError(f"{ckpt_path} has no per-fold confusion_matrix "
                       "(re-run training with the current scripts).")
    cm = np.zeros_like(np.array(fold_results[0]["confusion_matrix"]), dtype=float)
    for r in fold_results:
        cm += np.array(r["confusion_matrix"], dtype=float)
    return cm


def plot_panel(ax, cm, title, labels, show_ylabel):
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    n = len(labels)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(labels if show_ylabel else [""] * n, fontsize=8.5)
    ax.set_xlabel("Predicted", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("True", fontsize=8)
    ax.set_title(title, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = norm[i, j]
            on_diag = i == j
            ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1.0",
                    ha="center", va="center", fontsize=10,
                    color="white" if (on_diag or v > 0.5) else "black",
                    fontweight="bold" if on_diag else "normal")
    for s in GROUP_SEPS:
        ax.axhline(s, color="0.25", lw=1.1)
        ax.axvline(s, color="0.25", lw=1.1)
    return im


def main():
    p = argparse.ArgumentParser(description="Plot normalized confusion matrices")
    p.add_argument("--xgb", default=_latest("checkpoints/xgboost_fine_loso_cv_*.pkl"))
    p.add_argument("--bigru", default=_latest("checkpoints/bigru_fine_loso_cv_*.pkl"))
    p.add_argument("--unigru", default=_latest("checkpoints/stage2_A_fine_loso_cv_*.pkl"))
    p.add_argument("--out", default="figures/confusion_matrices")
    args = p.parse_args()

    panels = [
        (args.xgb, "(a) XGBoost + Summary"),
        (args.bigru, "(b) BiGRU + Temporal"),
        (args.unigru, "(c) UniGRU+CRF + Log Stitching"),
    ]
    for path, name in panels:
        print(f"  {name}: {path}")

    labels = [TASK_SHORT[i] for i in ORDER]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    im = None
    for ax, (path, title) in zip(axes, panels):
        cm = aggregated_cm(path)[np.ix_(ORDER, ORDER)]
        im = plot_panel(ax, cm, title, labels, show_ylabel=(ax is axes[0]))

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Row-normalized rate", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"  Saved {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
