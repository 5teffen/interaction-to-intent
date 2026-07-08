"""Shared training utilities for LOSO cross-validation."""

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np


def loso_splits(
    datasets: np.ndarray,
    train_sources: list[str],
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """
    Generate Leave-One-Source-Out splits.

    Each fold holds out one source for validation and trains on the rest.
    LOSO grouping is automatically correct: entire datasets are on one side
    of each split, so all users/sessions from a source stay together.

    Returns list of (train_idx, val_idx, held_out_source) tuples.
    """
    folds = []
    for held_out in train_sources:
        val_mask = datasets == held_out
        train_mask = np.isin(datasets, [s for s in train_sources if s != held_out])
        folds.append((np.where(train_mask)[0], np.where(val_mask)[0], held_out))
    return folds


def compute_median_epoch(best_epochs: list[int]) -> int:
    """Compute the median best epoch from CV folds, rounded to nearest int."""
    return int(np.median(best_epochs))


def decay_cumulative(sequences: list, indices: list, gamma: float) -> list:
    """
    Replace monotonic cumulative feature columns with an exponentially decayed
    (leaky) accumulator, so they reflect recent rather than whole-session
    accumulation. For each listed column c, recover per-step increments from the
    stored cumulative series and re-accumulate with decay:
        d_t = gamma * d_{t-1} + (C_t - C_{t-1}),   d_0 = C_0.
    gamma = 1.0 leaves the features unchanged (the original cumulative series).
    """
    if gamma >= 1.0 or not indices:
        return sequences
    out = []
    for s in sequences:
        s = np.asarray(s, dtype=float).copy()
        for c in indices:
            col = s[:, c]
            deltas = np.empty_like(col)
            deltas[0] = col[0]
            deltas[1:] = np.diff(col)
            acc = 0.0
            for t in range(len(col)):
                acc = gamma * acc + deltas[t]
                col[t] = acc
            s[:, c] = col
        out.append(s)
    return out


def standardize_per_source(X: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """
    Standardize summary-feature rows using statistics computed within each
    source separately (Option A). Each source is centered to zero mean and unit
    variance using only its own rows, which removes source-specific offsets that
    do not transfer across sources. Transductive and leakage-free: the held-out
    / test source self-normalizes from its own features, no labels involved.
    """
    X = np.asarray(X, dtype=float).copy()
    sources = np.asarray(sources)
    for s in np.unique(sources):
        m = sources == s
        mu = X[m].mean(axis=0)
        sd = X[m].std(axis=0)
        sd[sd < 1e-8] = 1.0
        X[m] = (X[m] - mu) / sd
    return X


def print_loso_summary(fold_results: list[dict], metric_keys: list[str]):
    """Print formatted LOSO CV summary with per-fold breakdown."""
    print(f"\n{'='*70}")
    print(f"  LOSO Cross-Validation Summary")
    print(f"{'='*70}")

    for key in metric_keys:
        values = [r[key] for r in fold_results if key in r]
        if values:
            print(f"  {key:25s}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    print(f"\n  Per-fold breakdown:")
    for r in fold_results:
        source = r.get("held_out_source", "?")
        parts = [f"{key}={r[key]:.4f}" for key in metric_keys if key in r]
        epoch_info = ""
        if "best_epoch" in r:
            epoch_info = f"  best_epoch={r['best_epoch']}"
        elif "best_n_estimators" in r:
            epoch_info = f"  best_n_estimators={r['best_n_estimators']}"
        size_info = f"  (train={r.get('n_train', '?')}, val={r.get('n_val', '?')})"
        print(f"    {source:12s}: {', '.join(parts)}{epoch_info}{size_info}")

    epoch_key = "best_epoch" if "best_epoch" in fold_results[0] else "best_n_estimators"
    if all(epoch_key in r for r in fold_results):
        epochs = [r[epoch_key] for r in fold_results]
        median_ep = compute_median_epoch(epochs)
        print(f"\n  {epoch_key}s: {epochs} -> median: {median_ep}")


def save_cv_checkpoint(path, fold_results: list[dict], model_config: dict,
                       extra: dict | None = None):
    """Save LOSO CV results to a pickle checkpoint."""
    epoch_key = "best_epoch" if "best_epoch" in fold_results[0] else "best_n_estimators"
    epochs = [r[epoch_key] for r in fold_results]

    ckpt = {
        "mode": "loso_cv",
        "fold_results": fold_results,
        "model_config": model_config,
        epoch_key + "s": epochs,
        "median_epoch": compute_median_epoch(epochs),
    }
    if extra:
        ckpt.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)
    print(f"  Saved LOSO CV checkpoint to {path}")


def load_cv_checkpoint(path) -> dict:
    """Load a LOSO CV checkpoint."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_test_epoch_log(
    log_entries: list[dict],
    model_name: str,
    output_dir: str | Path,
) -> Path:
    """Save per-epoch test metrics as a separate JSON file."""
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_epoch_log_{model_name}_{date_str}.json"
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(
            {
                "model": model_name,
                "description": (
                    "Per-epoch test set metrics (informational only, "
                    "not used for model selection)"
                ),
                "epochs": log_entries,
            },
            f,
            indent=2,
        )
    print(f"  Saved test epoch log to {path}")
    return path


def write_cv_json(path, fold_results: list[dict], model: str, label_scheme: str,
                  normalize: str, seed: int, ablation: dict | None = None,
                  metric_keys: list[str] | None = None):
    """Write a machine-readable LOSO CV summary (for ablation orchestration)."""
    if metric_keys is None:
        metric_keys = ["accuracy", "f1_macro", "f1_weighted"]
    summary = {}
    for k in metric_keys:
        vals = [r[k] for r in fold_results if k in r]
        if vals:
            summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # Aggregate the per-fold confusion matrices (folds partition the val data).
    agg_cm = None
    if fold_results and "confusion_matrix" in fold_results[0]:
        agg = np.array(fold_results[0]["confusion_matrix"])
        for r in fold_results[1:]:
            agg = agg + np.array(r["confusion_matrix"])
        agg_cm = agg.tolist()

    obj = {
        "model": model,
        "label_scheme": label_scheme,
        "normalize": normalize,
        "seed": seed,
        "ablation": ablation or {},
        "fold_results": fold_results,
        "summary": summary,
        "confusion_matrix_aggregated": agg_cm,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  Wrote CV summary JSON to {path}")


def save_test_results(output_dir, name: str, payload: dict):
    """Persist final held-out test metrics + confusion matrix as JSON."""
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"test_results_{name}_{date_str}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved test results to {path}")
    return path


def resolve_n_epochs(args, fold_results: list[dict] | None = None) -> int:
    """
    Determine the number of training epochs/iterations for full mode.

    Priority: --fixed-epochs/--fixed-iterations > --cv-checkpoint > fold_results from 'both' mode.
    """
    fixed = getattr(args, "fixed_epochs", None) or getattr(args, "fixed_iterations", None)
    if fixed is not None:
        return fixed

    if getattr(args, "cv_checkpoint", None):
        ckpt = load_cv_checkpoint(args.cv_checkpoint)
        return ckpt["median_epoch"]

    if fold_results is not None:
        epoch_key = "best_epoch" if "best_epoch" in fold_results[0] else "best_n_estimators"
        return compute_median_epoch([r[epoch_key] for r in fold_results])

    raise ValueError(
        "Full mode requires one of: --fixed-epochs/--fixed-iterations, "
        "--cv-checkpoint, or running in 'both' mode"
    )
