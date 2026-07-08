#!/usr/bin/env python3
"""
Evaluate trained models on the held-out test set.

Loads checkpoints from --checkpoint-dir, runs inference on test data,
and writes structured results to a JSON file for downstream analysis.
Also loads LOSO CV checkpoints when available to include per-fold
distributions alongside test results.

Usage:
    python scripts/evaluate.py \
        --checkpoint-dir checkpoints \
        --test-dir data/processed \
        --output data/processed/results.json
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
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from src.constants import N_CLASSES, TASK_LABEL_MAP, TASK_SHORT, get_label_scheme
from src.models.bigru import BiGRUClassifier, BiGRUAttentionClassifier
from src.models.stage2 import UniGRU_CRF
from src.datasets import (
    HoverSequenceDataset, TimestepLabelDataset,
    collate_sequences, collate_timesteps,
    normalize_sequences, normalize_sequences_per_source,
    normalize_sequences_per_session,
)
from src.training import standardize_per_source
from src.metrics import segment_f1, boundary_f1, metric_bundle


def _label_meta(ckpt: dict):
    """Per-checkpoint label metadata so coarse/fine checkpoints evaluate correctly."""
    task_map = ckpt.get("task_label_map") or TASK_LABEL_MAP
    task_map = {int(k): v for k, v in task_map.items()}
    n = len(task_map)
    names = [task_map[i] for i in range(n)]
    short = [task_map[i][:8] for i in range(n)]
    scheme = get_label_scheme(ckpt.get("label_scheme", "fine"))
    return task_map, n, names, short, scheme

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _find_checkpoint(ckpt_dir: Path, pattern: str,
                     exclude: str | None = None) -> Path | None:
    """Find the latest checkpoint matching a glob pattern.

    If *exclude* is given, skip paths whose stem contains that substring.
    """
    matches = sorted(
        (p for p in ckpt_dir.glob(pattern)
         if exclude is None or exclude not in p.stem),
        key=lambda p: p.stat().st_mtime,
    )
    return matches[-1] if matches else None


def _find_cv_checkpoint(ckpt_dir: Path, pattern: str,
                        exclude: str | None = None) -> dict | None:
    """Load the latest LOSO CV checkpoint matching a pattern."""
    path = _find_checkpoint(ckpt_dir, pattern, exclude=exclude)
    if path is None:
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _format_cv_results(cv_ckpt: dict | None) -> dict | None:
    """Extract LOSO fold results from a CV checkpoint for JSON output."""
    if cv_ckpt is None:
        return None
    fold_results = cv_ckpt.get("fold_results", [])
    if not fold_results:
        return None

    metric_keys = [k for k in fold_results[0] if k not in ("held_out_source", "n_train", "n_val")]
    summary = {}
    for key in metric_keys:
        values = [r[key] for r in fold_results if isinstance(r.get(key), (int, float))]
        if values:
            summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    return {
        "type": "loso_cv",
        "n_folds": len(fold_results),
        "per_fold": fold_results,
        "summary": summary,
        "median_epoch": cv_ckpt.get("median_epoch"),
    }


def evaluate_xgboost(ckpt_path: Path, test_data: dict,
                     cv_ckpt: dict | None = None) -> dict:
    """Evaluate XGBoost on test set."""
    logger.info(f"  XGBoost checkpoint: {ckpt_path.name}")
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    model = ckpt["model"]
    task_map, n_classes, names, short, scheme = _label_meta(ckpt)
    if ckpt.get("normalize") == "per-source":
        test_sources = np.array([m["dataset"] for m in test_data["metadata"]])
        X_test = standardize_per_source(test_data["features"], test_sources)
    else:
        X_test = ckpt["scaler"].transform(test_data["features"])
    y_test = scheme["remap"](test_data["labels"])

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_m = f1_score(y_test, y_pred, average="macro")
    f1_w = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))
    per_class = f1_score(y_test, y_pred, average=None, labels=list(range(n_classes)))

    report = classification_report(
        y_test, y_pred, target_names=names, output_dict=True, zero_division=0,
    )

    agreement = metric_bundle(y_test, y_pred, n_classes=n_classes, n_boot=2000)
    logger.info(f"    Accuracy: {acc:.4f}  F1-macro: {f1_m:.4f}  F1-weighted: {f1_w:.4f}")
    logger.info(f"    MCC: {agreement['mcc']['value']:.4f}  kappa: {agreement['kappa']['value']:.4f}")

    importance = None
    if hasattr(model, "feature_importances_"):
        importance = {
            name: float(v)
            for name, v in zip(ckpt.get("feature_names", []), model.feature_importances_)
        }

    return {
        "model": "XGBoost",
        "stage": 1,
        "label_scheme": ckpt.get("label_scheme", "fine"),
        "accuracy": float(acc),
        "f1_macro": float(f1_m),
        "f1_weighted": float(f1_w),
        "per_class_f1": {short[i]: float(per_class[i]) for i in range(n_classes)},
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "agreement": agreement,
        "feature_importance": importance,
        "loso_cv": _format_cv_results(cv_ckpt),
    }


def evaluate_bigru(ckpt_path: Path, test_data: dict, device: torch.device,
                   cv_ckpt: dict | None = None) -> dict:
    """Evaluate BiGRU on test set."""
    logger.info(f"  BiGRU checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ckpt["model_config"]
    task_map, n_classes, names, short, scheme = _label_meta(ckpt)
    attention = cfg.get("attention", False)
    ModelClass = BiGRUAttentionClassifier if attention else BiGRUClassifier
    model = ModelClass(
        input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
        num_classes=cfg.get("num_classes", n_classes), dropout=cfg.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if ckpt.get("normalize") == "per-source":
        test_sources = np.array([m["dataset"] for m in test_data["metadata"]])
        test_seqs = normalize_sequences_per_source(test_data["sequences"], test_sources)
    else:
        mean, std = ckpt["feature_mean"], ckpt["feature_std"]
        test_seqs = normalize_sequences(test_data["sequences"], mean, std)
    y_test = scheme["remap"](test_data["labels"])

    dataset = HoverSequenceDataset(test_seqs, y_test)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, collate_fn=collate_sequences,
    )

    all_preds, all_true = [], []
    with torch.no_grad():
        for x, y, lengths in loader:
            x, lengths = x.to(device), lengths.to(device)
            logits = model(x, lengths)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_true.append(y.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    acc = accuracy_score(y_true, y_pred)
    f1_m = f1_score(y_true, y_pred, average="macro")
    f1_w = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    per_class = f1_score(y_true, y_pred, average=None, labels=list(range(n_classes)))

    variant_name = "BiGRU+Attn" if attention else "BiGRU"
    agreement = metric_bundle(y_true, y_pred, n_classes=n_classes, n_boot=2000)
    logger.info(f"    Accuracy: {acc:.4f}  F1-macro: {f1_m:.4f}  F1-weighted: {f1_w:.4f}")
    logger.info(f"    MCC: {agreement['mcc']['value']:.4f}  kappa: {agreement['kappa']['value']:.4f}")

    return {
        "model": variant_name,
        "stage": 1,
        "label_scheme": ckpt.get("label_scheme", "fine"),
        "accuracy": float(acc),
        "f1_macro": float(f1_m),
        "f1_weighted": float(f1_w),
        "per_class_f1": {short[i]: float(per_class[i]) for i in range(n_classes)},
        "confusion_matrix": cm.tolist(),
        "agreement": agreement,
        "loso_cv": _format_cv_results(cv_ckpt),
    }


def evaluate_stage2(
    ckpt_path: Path, test_data: dict, device: torch.device,
    boundary_tolerance: int = 3, cv_ckpt: dict | None = None,
) -> dict:
    """Evaluate Stage 2 UniGRU+CRF on test sequences."""
    logger.info(f"  Stage 2 checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ckpt["model_config"]
    task_map, n_classes, names, short, scheme = _label_meta(ckpt)
    model = UniGRU_CRF(
        input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
        num_classes=cfg.get("num_classes", n_classes), dropout=cfg.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if ckpt.get("normalize") == "per-source":
        meta = test_data.get("metadata")
        if meta is not None and "dataset" in meta[0]:
            test_sources = np.array([m["dataset"] for m in meta])
            test_seqs = normalize_sequences_per_source(test_data["sequences"], test_sources)
        else:
            test_seqs = normalize_sequences_per_session(test_data["sequences"])
    else:
        mean, std = ckpt["feature_mean"], ckpt["feature_std"]
        test_seqs = normalize_sequences(test_data["sequences"], mean, std)
    test_labels = [scheme["remap"](l) for l in test_data["labels"]]

    dataset = TimestepLabelDataset(test_seqs, test_labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=collate_timesteps,
    )

    all_preds_list, all_true_list = [], []
    with torch.no_grad():
        for x, y, lengths in loader:
            x, lengths = x.to(device), lengths.to(device)
            decoded = model.decode(x, lengths)
            for i, length in enumerate(lengths):
                l = length.item()
                all_preds_list.append(np.array(decoded[i][:l]))
                all_true_list.append(y[i][:l].numpy())

    preds_flat = np.concatenate(all_preds_list)
    true_flat = np.concatenate(all_true_list)

    acc = accuracy_score(true_flat, preds_flat)
    f1_m = f1_score(true_flat, preds_flat, average="macro")
    f1_w = f1_score(true_flat, preds_flat, average="weighted")
    cm = confusion_matrix(true_flat, preds_flat, labels=list(range(n_classes)))
    per_class = f1_score(true_flat, preds_flat, average=None, labels=list(range(n_classes)))

    seg_f1 = segment_f1(all_true_list, all_preds_list, n_classes=n_classes)
    bnd = boundary_f1(all_true_list, all_preds_list, tolerance=boundary_tolerance)
    agreement = metric_bundle(true_flat, preds_flat, n_classes=n_classes, n_boot=2000)

    variant = ckpt.get("variant", "unknown")
    logger.info(f"    Variant: {variant}")
    logger.info(f"    TS Accuracy: {acc:.4f}  TS F1-macro: {f1_m:.4f}")
    logger.info(f"    Segment F1:  {seg_f1:.4f}  Boundary F1: {bnd['f1']:.4f}")
    logger.info(f"    MCC: {agreement['mcc']['value']:.4f}  kappa: {agreement['kappa']['value']:.4f}")

    return {
        "model": "UniGRU+CRF",
        "stage": 2,
        "variant": variant,
        "label_scheme": ckpt.get("label_scheme", "fine"),
        "ts_accuracy": float(acc),
        "ts_f1_macro": float(f1_m),
        "ts_f1_weighted": float(f1_w),
        "per_class_f1": {short[i]: float(per_class[i]) for i in range(n_classes)},
        "confusion_matrix": cm.tolist(),
        "segment_f1": float(seg_f1),
        "boundary_precision": float(bnd["precision"]),
        "boundary_recall": float(bnd["recall"]),
        "boundary_f1": float(bnd["f1"]),
        "agreement": agreement,
        "loso_cv": _format_cv_results(cv_ckpt),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on held-out test set")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--test-dir", required=True, help="Directory with test .pkl files")
    parser.add_argument("--output", required=True, help="Output results JSON path")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--boundary-tolerance", type=int, default=3)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    test_dir = Path(args.test_dir)
    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info(f"Device: {device}")

    results = {"task_labels": TASK_LABEL_MAP, "task_short": TASK_SHORT, "models": []}

    # Stage 1: XGBoost
    xgb_ckpt = _find_checkpoint(ckpt_dir, "stage1_xgboost_*.pkl")
    xgb_cv = _find_cv_checkpoint(ckpt_dir, "xgboost_*loso_cv_*.pkl")
    test_summary_path = test_dir / "test_summary.pkl"
    if xgb_ckpt and test_summary_path.exists():
        with open(test_summary_path, "rb") as f:
            test_summary = pickle.load(f)
        results["models"].append(evaluate_xgboost(xgb_ckpt, test_summary, xgb_cv))

    # Stage 1: BiGRU / BiGRU+Attn (share the same test data)
    test_temporal_path = test_dir / "test_temporal.pkl"
    test_temporal = None
    if test_temporal_path.exists():
        with open(test_temporal_path, "rb") as f:
            test_temporal = pickle.load(f)

    bigru_ckpt = _find_checkpoint(ckpt_dir, "stage1_bigru_*.pt", exclude="_attn_")
    bigru_cv = _find_cv_checkpoint(ckpt_dir, "bigru_*loso_cv_*.pkl", exclude="_attn_")
    if bigru_ckpt and test_temporal is not None:
        results["models"].append(
            evaluate_bigru(bigru_ckpt, test_temporal, device, bigru_cv)
        )

    bigru_attn_ckpt = _find_checkpoint(ckpt_dir, "stage1_bigru_*_attn_*.pt")
    bigru_attn_cv = _find_cv_checkpoint(ckpt_dir, "bigru_*_attn_*loso_cv_*.pkl")
    if bigru_attn_ckpt and test_temporal is not None:
        results["models"].append(
            evaluate_bigru(bigru_attn_ckpt, test_temporal, device, bigru_attn_cv)
        )

    # Stage 2: Variant A
    s2a_ckpt = _find_checkpoint(ckpt_dir, "stage2_A_*unigru*.pt")
    s2a_cv = _find_cv_checkpoint(ckpt_dir, "stage2_A_*loso_cv_*.pkl")
    test_s2a_path = test_dir / "test_stage2_variantA.pkl"
    if s2a_ckpt and test_s2a_path.exists():
        with open(test_s2a_path, "rb") as f:
            test_s2a = pickle.load(f)
        results["models"].append(
            evaluate_stage2(s2a_ckpt, test_s2a, device, args.boundary_tolerance, s2a_cv)
        )

    # Stage 2: Variant B
    s2b_ckpt = _find_checkpoint(ckpt_dir, "stage2_B_*unigru*.pt")
    s2b_cv = _find_cv_checkpoint(ckpt_dir, "stage2_B_*loso_cv_*.pkl")
    test_s2b_path = test_dir / "test_stage2_variantB.pkl"
    if s2b_ckpt and test_s2b_path.exists():
        with open(test_s2b_path, "rb") as f:
            test_s2b = pickle.load(f)
        results["models"].append(
            evaluate_stage2(s2b_ckpt, test_s2b, device, args.boundary_tolerance, s2b_cv)
        )

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"Evaluated {len(results['models'])} models")


if __name__ == "__main__":
    main()
