#!/usr/bin/env python3
"""
Prepare Stage 1 features (temporal + summary) for train/test splits.

Splits data-filtered.csv by dataset name, then featurizes both splits.
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

from src.constants import TASK_LABEL_MAP, TEMPORAL_FEATURE_NAMES, SUMMARY_FEATURE_NAMES
from src.data import load_projected_data, precompute_spatial_stats
from src.features.temporal import featurize_sequence
from src.features.summary import featurize_sequence_summary

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def split_data(data_csv: str, test_datasets: str):
    df = pd.read_csv(data_csv)
    test_names = set(test_datasets.split(","))
    test_df = df[df["dataset"].isin(test_names)].reset_index(drop=True)
    train_df = df[~df["dataset"].isin(test_names)].reset_index(drop=True)
    logger.info(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    return train_df, test_df


def featurize_temporal_split(df, proj_data, centroids_cache, density_cache):
    sequences, labels, metadata = [], [], []

    for idx, row in df.iterrows():
        events = json.loads(row["sequence_json"])
        key = (row["dataset"], row["projection"])
        if key not in proj_data:
            logger.warning(f"Row {idx}: No projected CSV for {key}, skipping")
            continue

        feat = featurize_sequence(
            events, proj_data[key], centroids_cache[key], density_cache[key],
        )
        sequences.append(feat)
        labels.append(int(row["task_type"]))
        metadata.append({
            "dataset": row["dataset"], "projection": row["projection"],
            "user": row["prolific_id"], "reward": row["reward"],
        })

        if (idx + 1) % 200 == 0:
            logger.info(f"  Temporal: {idx + 1}/{len(df)}")

    return {
        "sequences": sequences,
        "labels": np.array(labels, dtype=np.int64),
        "metadata": metadata,
        "feature_names": TEMPORAL_FEATURE_NAMES,
        "task_label_map": TASK_LABEL_MAP,
    }


def featurize_summary_split(df, proj_data, centroids_cache, density_cache):
    features_list, labels, metadata = [], [], []

    for idx, row in df.iterrows():
        events = json.loads(row["sequence_json"])
        key = (row["dataset"], row["projection"])
        if key not in proj_data:
            logger.warning(f"Row {idx}: No projected CSV for {key}, skipping")
            continue

        feat = featurize_sequence_summary(
            events, proj_data[key], centroids_cache[key], density_cache[key],
        )
        features_list.append(feat)
        labels.append(int(row["task_type"]))
        metadata.append({
            "dataset": row["dataset"], "projection": row["projection"],
            "user": row["prolific_id"], "reward": row["reward"],
        })

        if (idx + 1) % 200 == 0:
            logger.info(f"  Summary: {idx + 1}/{len(df)}")

    if not features_list:
        features = np.empty((0, len(SUMMARY_FEATURE_NAMES)), dtype=np.float32)
    else:
        features = np.stack(features_list).astype(np.float32)

    return {
        "features": features,
        "labels": np.array(labels, dtype=np.int64),
        "metadata": metadata,
        "feature_names": SUMMARY_FEATURE_NAMES,
        "task_label_map": TASK_LABEL_MAP,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Stage 1 features")
    parser.add_argument("--data-csv", default=None,
                        help="Combined CSV to split (legacy / reference)")
    parser.add_argument("--train-csv", default=None,
                        help="Pre-split training CSV (use instead of --data-csv)")
    parser.add_argument("--test-csv", default=None,
                        help="Pre-split test CSV (use instead of --data-csv)")
    parser.add_argument("--proj-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-datasets", default="recipes")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_csv and args.test_csv:
        train_df = pd.read_csv(args.train_csv)
        test_df = pd.read_csv(args.test_csv)
        logger.info(f"Loaded pre-split data: Train={len(train_df)}, Test={len(test_df)}")
    elif args.data_csv:
        train_df, test_df = split_data(args.data_csv, args.test_datasets)
        train_df.to_csv(output_dir / "train_filtered.csv", index=False)
        if len(test_df) > 0:
            test_df.to_csv(output_dir / "test_filtered.csv", index=False)
    else:
        parser.error("Provide either --data-csv or both --train-csv and --test-csv")

    proj_data = load_projected_data(Path(args.proj_dir))
    centroids_cache, density_cache = precompute_spatial_stats(proj_data)

    for split_name, split_df in [("train", train_df), ("test", test_df)]:
        if len(split_df) == 0:
            logger.info(f"Skipping {split_name} (empty)")
            continue

        logger.info(f"Featurizing {split_name} temporal ({len(split_df)} seqs)...")
        temporal = featurize_temporal_split(split_df, proj_data, centroids_cache, density_cache)
        with open(output_dir / f"{split_name}_temporal.pkl", "wb") as f:
            pickle.dump(temporal, f)

        logger.info(f"Featurizing {split_name} summary ({len(split_df)} seqs)...")
        summary = featurize_summary_split(split_df, proj_data, centroids_cache, density_cache)
        with open(output_dir / f"{split_name}_summary.pkl", "wb") as f:
            pickle.dump(summary, f)

    logger.info("Feature preparation complete!")


if __name__ == "__main__":
    main()
