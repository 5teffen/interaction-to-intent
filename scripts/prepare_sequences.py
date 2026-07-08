#!/usr/bin/env python3
"""
Prepare Stage 2 composed sequences (Variant A and Variant B).

Variant A: Raw concatenation of events within same (dataset, projection), 24-dim.
           Outputs per-source pkl files for LOSO cross-validation.
Variant B: Feature stitching across any sequences, 15-dim stateless.
           Outputs atomics pkl for on-the-fly composition during LOSO.

Usage:
    python scripts/prepare_sequences.py \
        --data-csv data/processed/train_filtered.csv \
        --proj-dir data/projected \
        --output-dir data/processed \
        --n-sessions 2000 --seed 42

    # With test set:
    python scripts/prepare_sequences.py \
        --data-csv data/processed/train_filtered.csv \
        --test-data-csv data/processed/test_filtered.csv \
        --proj-dir data/projected \
        --output-dir data/processed
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.constants import (
    TASK_LABEL_MAP, TEMPORAL_FEATURE_NAMES, STATELESS_FEATURE_NAMES,
)
from src.data import load_projected_data, precompute_spatial_stats
from src.features.temporal import featurize_sequence
from src.features.stateless import featurize_stateless
from src.compose import (
    compose_session_events, compose_session_labels,
    stitch_feature_matrices,
    generate_sessions_same_projection, generate_sessions_any,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ── Variant A helpers ────────────────────────────────────────────────────────

def _compose_variant_a(
    df, n_sessions, min_tasks, max_tasks, seed,
    proj_data, centroids_cache, density_cache, source_label=None,
):
    """Compose Variant A sessions from a DataFrame of atomic sequences."""
    rng = np.random.default_rng(seed)

    parsed_events = {}
    for idx, row in df.iterrows():
        parsed_events[idx] = json.loads(row["sequence_json"])

    session_specs = generate_sessions_same_projection(
        df, n_sessions, (min_tasks, max_tasks), rng,
    )

    sequences, labels, metadata = [], [], []
    for i, spec in enumerate(session_specs):
        key = (spec["dataset"], spec["projection"])
        if key not in proj_data:
            continue

        atomic_seqs = [parsed_events[idx] for idx in spec["atomic_indices"]]
        composed = compose_session_events(atomic_seqs)
        feat = featurize_sequence(
            composed, proj_data[key], centroids_cache[key], density_cache[key],
        )
        ts_labels = compose_session_labels(atomic_seqs, spec["task_types"])

        sequences.append(feat)
        labels.append(ts_labels)
        metadata.append({
            "dataset": spec["dataset"], "projection": spec["projection"],
            "n_tasks": len(spec["task_types"]),
            "task_sequence": spec["task_types"],
            "atomic_indices": spec["atomic_indices"],
        })
        if (i + 1) % 500 == 0:
            tag = f" [{source_label}]" if source_label else ""
            logger.info(f"  Variant A{tag}: {i + 1}/{n_sessions}")

    result = {
        "sequences": sequences,
        "labels": labels,
        "metadata": metadata,
        "feature_names": TEMPORAL_FEATURE_NAMES,
        "task_label_map": TASK_LABEL_MAP,
        "variant": "A_raw_concatenation",
        "config": {
            "n_sessions": len(sequences), "min_tasks": min_tasks,
            "max_tasks": max_tasks, "seed": seed,
        },
    }
    if source_label:
        result["source"] = source_label
    return result


# ── Variant B helpers ────────────────────────────────────────────────────────

def _featurize_variant_b_atomics(df, proj_data, centroids_cache, density_cache):
    """Pre-featurize all atomic sequences for Variant B, retaining per-atom metadata."""
    sequences, labels_list, metadata = [], [], []

    for idx, row in df.iterrows():
        events = json.loads(row["sequence_json"])
        key = (row["dataset"], row["projection"])
        if key not in proj_data:
            continue
        feat = featurize_stateless(
            events, proj_data[key], centroids_cache[key], density_cache[key],
        )
        sequences.append(feat)
        labels_list.append(int(row["task_type"]))
        metadata.append({
            "dataset": row["dataset"],
            "projection": row["projection"],
            "user": row["prolific_id"],
        })
        if (idx + 1) % 200 == 0:
            logger.info(f"  Variant B atomics: {idx + 1}/{len(df)}")

    return {
        "sequences": sequences,
        "labels": labels_list,
        "metadata": metadata,
        "feature_names": STATELESS_FEATURE_NAMES,
        "task_label_map": TASK_LABEL_MAP,
        "type": "atomics",
    }


def _compose_variant_b(
    df, n_sessions, min_tasks, max_tasks, seed,
    proj_data, centroids_cache, density_cache,
):
    """Compose Variant B sessions (used for test set output)."""
    rng = np.random.default_rng(seed)

    atomic_features, atomic_labels = [], []
    for idx, row in df.iterrows():
        events = json.loads(row["sequence_json"])
        key = (row["dataset"], row["projection"])
        if key not in proj_data:
            continue
        feat = featurize_stateless(
            events, proj_data[key], centroids_cache[key], density_cache[key],
        )
        atomic_features.append(feat)
        atomic_labels.append(int(row["task_type"]))

    session_indices = generate_sessions_any(
        len(atomic_features), n_sessions, (min_tasks, max_tasks), rng,
    )

    sequences, labels, metadata = [], [], []
    for i, indices in enumerate(session_indices):
        matrices = [atomic_features[idx] for idx in indices]
        task_types = [atomic_labels[idx] for idx in indices]

        stitched = stitch_feature_matrices(matrices)
        ts_labels = np.concatenate(
            [[t] * m.shape[0] for m, t in zip(matrices, task_types)]
        ).astype(np.int64)

        sequences.append(stitched)
        labels.append(ts_labels)
        metadata.append({
            "n_tasks": len(task_types),
            "task_sequence": task_types,
            "atomic_indices": indices,
        })

    return {
        "sequences": sequences,
        "labels": labels,
        "metadata": metadata,
        "feature_names": STATELESS_FEATURE_NAMES,
        "task_label_map": TASK_LABEL_MAP,
        "variant": "B_feature_stitching",
        "config": {
            "n_sessions": len(sequences), "min_tasks": min_tasks,
            "max_tasks": max_tasks, "seed": seed,
        },
    }


# ── Matched test set (identical sessions for both variants) ──────────────────

def _compose_matched_test(
    df, n_sessions, min_tasks, max_tasks, seed,
    proj_data, centroids_cache, density_cache,
):
    """
    Build Variant A and Variant B test sets from ONE shared set of session
    specs, so the two test sets contain exactly the same multi-task sessions
    (same atomics, order, and per-timestep labels) and differ only in the
    representation/stitching method. Sessions are same-projection (required for
    Variant A to be valid); Variant B's cross-projection capability is a
    separate qualitative claim and is not exercised here on purpose, to keep the
    A-vs-B accuracy comparison apples-to-apples.
    """
    rng = np.random.default_rng(seed)
    parsed_events = {idx: json.loads(row["sequence_json"]) for idx, row in df.iterrows()}
    specs = generate_sessions_same_projection(df, n_sessions, (min_tasks, max_tasks), rng)

    seqs_a, seqs_b, labels, metadata = [], [], [], []
    skipped = 0
    for i, spec in enumerate(specs):
        key = (spec["dataset"], spec["projection"])
        if key not in proj_data:
            skipped += 1
            continue
        atomic_seqs = [parsed_events[idx] for idx in spec["atomic_indices"]]

        composed = compose_session_events(atomic_seqs)
        feat_a = featurize_sequence(
            composed, proj_data[key], centroids_cache[key], density_cache[key],
        )
        mats = [
            featurize_stateless(ev, proj_data[key], centroids_cache[key], density_cache[key])
            for ev in atomic_seqs
        ]
        feat_b = stitch_feature_matrices(mats)
        ts_labels = compose_session_labels(atomic_seqs, spec["task_types"])

        if not (len(feat_a) == len(feat_b) == len(ts_labels)):
            skipped += 1
            continue

        seqs_a.append(feat_a)
        seqs_b.append(feat_b)
        labels.append(ts_labels)
        metadata.append({
            "dataset": spec["dataset"], "projection": spec["projection"],
            "n_tasks": len(spec["task_types"]),
            "task_sequence": spec["task_types"],
            "atomic_indices": spec["atomic_indices"],
        })
        if (i + 1) % 500 == 0:
            logger.info(f"  Matched test: {i + 1}/{n_sessions}")

    if skipped:
        logger.warning(f"  Matched test: skipped {skipped} sessions (missing proj / length mismatch)")

    cfg = {"n_sessions": len(seqs_a), "min_tasks": min_tasks,
           "max_tasks": max_tasks, "seed": seed, "matched": True}
    result_a = {
        "sequences": seqs_a, "labels": labels, "metadata": metadata,
        "feature_names": TEMPORAL_FEATURE_NAMES, "task_label_map": TASK_LABEL_MAP,
        "variant": "A_raw_concatenation", "config": cfg,
    }
    result_b = {
        "sequences": seqs_b, "labels": labels, "metadata": metadata,
        "feature_names": STATELESS_FEATURE_NAMES, "task_label_map": TASK_LABEL_MAP,
        "variant": "B_feature_stitching", "config": cfg,
    }
    return result_a, result_b


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Stage 2 composed sequences (Variant A + B)"
    )
    parser.add_argument("--data-csv", required=True, help="Path to train CSV")
    parser.add_argument("--proj-dir", required=True, help="Directory with projected CSVs")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--n-sessions", type=int, default=2000,
                        help="Sessions per source (Variant A) or total test sessions")
    parser.add_argument("--min-tasks", type=int, default=2)
    parser.add_argument("--max-tasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-data-csv", type=str, default=None,
                        help="Optional test CSV for Stage 2 test sequences")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proj_data = load_projected_data(Path(args.proj_dir))
    centroids_cache, density_cache = precompute_spatial_stats(proj_data)

    train_df = pd.read_csv(args.data_csv)
    sources = sorted(train_df["dataset"].unique())

    # ── Variant A: per-source composed sessions ──────────────────────────
    for source in sources:
        logger.info(f"Preparing Variant A for source '{source}'...")
        source_df = train_df[train_df["dataset"] == source].reset_index(drop=True)
        if len(source_df) == 0:
            logger.warning(f"  No data for source '{source}', skipping")
            continue

        result = _compose_variant_a(
            source_df, args.n_sessions,
            args.min_tasks, args.max_tasks, args.seed,
            proj_data, centroids_cache, density_cache,
            source_label=source,
        )
        out_path = output_dir / f"stage2_varA_{source}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        logger.info(f"  Saved: {out_path} ({len(result['sequences'])} sessions)")

    # ── Variant B: atomics with per-sequence metadata ────────────────────
    logger.info("Preparing Variant B atomics...")
    atomics = _featurize_variant_b_atomics(
        train_df, proj_data, centroids_cache, density_cache,
    )
    out_path = output_dir / "stage2_varB_atomics.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(atomics, f)
    logger.info(f"  Saved: {out_path} ({len(atomics['sequences'])} atomic sequences)")

    # ── Test sequences (if provided) ─────────────────────────────────────
    if args.test_data_csv:
        test_csv = Path(args.test_data_csv)
        if not test_csv.is_file():
            logger.error(f"Test CSV not found: {test_csv}")
        else:
            test_df = pd.read_csv(test_csv)

            logger.info("Preparing matched test sessions (identical for A and B)...")
            test_a, test_b = _compose_matched_test(
                test_df, args.n_sessions,
                args.min_tasks, args.max_tasks, args.seed + 1,
                proj_data, centroids_cache, density_cache,
            )
            with open(output_dir / "test_stage2_variantA.pkl", "wb") as f:
                pickle.dump(test_a, f)
            with open(output_dir / "test_stage2_variantB.pkl", "wb") as f:
                pickle.dump(test_b, f)
            logger.info(
                f"  Saved matched test sets: {len(test_a['sequences'])} sessions each "
                f"(A=24-dim stateful, B=15-dim stateless, identical sessions/labels)"
            )

    logger.info("Sequence preparation complete!")


if __name__ == "__main__":
    main()
