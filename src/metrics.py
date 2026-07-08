"""Evaluation metrics for Stage 2 per-timestep prediction."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score,
)


# ── Chance-corrected agreement + bootstrap CIs (Stage 1) ─────────────────────

def agreement_scores(y_true, y_pred) -> dict:
    """Chance-corrected agreement: Matthews correlation and Cohen's kappa."""
    return {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def _metric_fns(n_classes: int | None):
    """Metric name -> callable(y_true, y_pred). Robust to absent classes."""
    labels = list(range(n_classes)) if n_classes else None
    return {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "f1_macro": lambda a, b: f1_score(a, b, average="macro",
                                          labels=labels, zero_division=0),
        "mcc": matthews_corrcoef,
        "kappa": cohen_kappa_score,
    }


def metric_bundle(y_true, y_pred, n_classes: int | None = None,
                  n_boot: int = 0, seed: int = 0, alpha: float = 0.05) -> dict:
    """
    Point estimates for accuracy, balanced accuracy, macro-F1, MCC, kappa.
    If n_boot > 0, attach percentile bootstrap CIs (resampling prediction pairs).
    Returns {metric: {"value": v, "ci": [lo, hi]}}.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fns = _metric_fns(n_classes)

    boot_stats = {k: [] for k in fns} if n_boot else None
    if n_boot:
        rng = np.random.default_rng(seed)
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yt, yp = y_true[idx], y_pred[idx]
            for k, fn in fns.items():
                boot_stats[k].append(fn(yt, yp))

    out = {}
    for k, fn in fns.items():
        entry = {"value": float(fn(y_true, y_pred))}
        if n_boot:
            lo, hi = np.percentile(boot_stats[k], [100 * alpha / 2, 100 * (1 - alpha / 2)])
            entry["ci"] = [float(lo), float(hi)]
        out[k] = entry
    return out


def format_bundle(bundle: dict, keys=("mcc", "kappa")) -> str:
    """One-line printable summary, e.g. 'MCC=0.512 [0.45, 0.58]  kappa=...'."""
    parts = []
    for k in keys:
        if k not in bundle:
            continue
        e = bundle[k]
        if "ci" in e:
            parts.append(f"{k}={e['value']:.4f} [{e['ci'][0]:.4f}, {e['ci'][1]:.4f}]")
        else:
            parts.append(f"{k}={e['value']:.4f}")
    return "  ".join(parts)


def segment_f1(
    true_list: list[np.ndarray],
    pred_list: list[np.ndarray],
    n_classes: int | None = None,
) -> float:
    """
    Segment-level F1: fraction of ground-truth contiguous segments where
    the majority prediction matches the true label.
    """
    correct = 0
    total = 0
    for true, pred in zip(true_list, pred_list):
        changes = np.where(np.diff(true) != 0)[0] + 1
        segments = np.split(np.arange(len(true)), changes)
        minlen = n_classes if n_classes is not None else int(pred.max()) + 1
        for seg_indices in segments:
            if len(seg_indices) == 0:
                continue
            true_label = true[seg_indices[0]]
            pred_majority = np.bincount(pred[seg_indices], minlength=minlen).argmax()
            correct += int(pred_majority == true_label)
            total += 1
    return correct / total if total > 0 else 0.0


def boundary_f1(
    true_list: list[np.ndarray],
    pred_list: list[np.ndarray],
    tolerance: int = 3,
) -> dict[str, float]:
    """
    Boundary detection F1: does the model detect task transitions within
    +/- tolerance timesteps?

    Returns dict with keys: precision, recall, f1.
    """
    tp, fp, fn = 0, 0, 0
    for true, pred in zip(true_list, pred_list):
        true_boundaries = set(np.where(np.diff(true) != 0)[0] + 1)
        pred_boundaries = set(np.where(np.diff(pred) != 0)[0] + 1)

        matched_pred: set[int] = set()
        for tb in true_boundaries:
            candidates = sorted(
                (pb for pb in pred_boundaries
                 if abs(tb - pb) <= tolerance and pb not in matched_pred),
                key=lambda pb: abs(tb - pb),
            )
            if candidates:
                matched_pred.add(candidates[0])
                tp += 1
            else:
                fn += 1
        fp += len(pred_boundaries) - len(matched_pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
