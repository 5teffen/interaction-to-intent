#!/usr/bin/env python3
"""
Stage 2 dataset summary: how multi-task sessions are composed.

Four panels over the composed Variant A sessions (per-source pkls):
    (a) tasks per session         (uniform 2-5 by construction)
    (b) session length (timesteps)
    (c) per-task frequency        (how often each atomic task appears)
    (d) task-to-task adjacency    (P(next | current); ~uniform = random ordering)

Usage:
    python scripts/plot_session_composition.py \
        --data-dir data/processed --out figures/session_composition
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

from src.constants import TASK_SHORT, N_CLASSES


def collect(data_dir):
    files = sorted(glob.glob(str(Path(data_dir) / "stage2_varA_*.pkl")))
    n_tasks, lengths = [], []
    freq = np.zeros(N_CLASSES, dtype=int)
    adj = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for f in files:
        d = pickle.load(open(f, "rb"))
        for seq, m in zip(d["sequences"], d["metadata"]):
            ts = [int(t) for t in m["task_sequence"]]
            n_tasks.append(len(ts))
            lengths.append(seq.shape[0])
            for t in ts:
                freq[t] += 1
            for a, b in zip(ts[:-1], ts[1:]):
                adj[a, b] += 1
    return np.array(n_tasks), np.array(lengths), freq, adj, len(n_tasks)


def main():
    p = argparse.ArgumentParser(description="Stage 2 session-composition summary")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out", default="figures/session_composition")
    args = p.parse_args()

    n_tasks, lengths, freq, adj, total = collect(args.data_dir)
    print(f"  {total} composed sessions")

    fig, ax = plt.subplots(2, 2, figsize=(8.2, 6.4), constrained_layout=True)
    blue = "#4C72B0"

    # (a) tasks per session
    vals, counts = np.unique(n_tasks, return_counts=True)
    ax[0, 0].bar(vals, counts, color=blue, width=0.7)
    ax[0, 0].set_xlabel("Tasks per session")
    ax[0, 0].set_ylabel("Sessions")
    ax[0, 0].set_xticks(vals)
    ax[0, 0].set_title("(a) Tasks per session", fontsize=10)

    # (b) session length
    ax[0, 1].hist(lengths, bins=40, color=blue)
    ax[0, 1].axvline(np.median(lengths), color="black", ls="--", lw=1,
                     label=f"median {int(np.median(lengths))}")
    ax[0, 1].set_xlabel("Session length (timesteps)")
    ax[0, 1].set_ylabel("Sessions")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title("(b) Session length", fontsize=10)

    # (c) per-task frequency
    ax[1, 0].bar(range(N_CLASSES), freq, color=blue, width=0.7)
    ax[1, 0].set_xticks(range(N_CLASSES))
    ax[1, 0].set_xticklabels(TASK_SHORT, rotation=45, ha="right", fontsize=7)
    ax[1, 0].set_ylabel("Occurrences")
    ax[1, 0].set_title("(c) Per-task frequency", fontsize=10)

    # (d) adjacency P(next | current)
    row = adj.sum(axis=1, keepdims=True)
    prob = np.divide(adj, row, out=np.zeros_like(adj, dtype=float), where=row > 0)
    im = ax[1, 1].imshow(prob, cmap="Blues", vmin=0, vmax=prob.max())
    ax[1, 1].set_xticks(range(N_CLASSES)); ax[1, 1].set_yticks(range(N_CLASSES))
    ax[1, 1].set_xticklabels(TASK_SHORT, rotation=45, ha="right", fontsize=6)
    ax[1, 1].set_yticklabels(TASK_SHORT, fontsize=6)
    ax[1, 1].set_xlabel("Next task"); ax[1, 1].set_ylabel("Current task")
    ax[1, 1].set_title("(d) Task transitions $P(\\mathrm{next}\\mid\\mathrm{current})$", fontsize=10)
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.04)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"  Saved {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
