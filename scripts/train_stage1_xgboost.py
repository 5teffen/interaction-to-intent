"""
Stage 1: XGBoost on Summary Features
=====================================
Sequence-level classification using fixed-length summary feature vectors.

Modes:
    cv    — LOSO cross-validation only (default)
    full  — Refit on all training data, evaluate on test
    both  — CV then full (uses median best_n_estimators from CV)

Usage:
    # LOSO cross-validation
    python train_stage1_xgboost.py --data train_summary.pkl --save-model checkpoints/

    # Full training from CV checkpoint
    python train_stage1_xgboost.py --data train_summary.pkl --mode full \
        --test-data test_summary.pkl --cv-checkpoint checkpoints/xgboost_loso_cv.pkl \
        --save-model checkpoints/

    # Both in one run
    python train_stage1_xgboost.py --data train_summary.pkl --mode both \
        --test-data test_summary.pkl --save-model checkpoints/
"""

import argparse
import json
import pickle
import sys
import numpy as np
from collections import Counter
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

import src.constants as C
from src.constants import N_CLASSES, TASK_SHORT, TASK_LABEL_MAP, TRAIN_DATASETS
from src.constants import (
    get_label_scheme, select_feature_indices, resolve_group_features,
    SUMMARY_FEATURE_GROUPS,
)
from src.training import (
    loso_splits, print_loso_summary, save_cv_checkpoint,
    save_test_epoch_log, resolve_n_epochs, standardize_per_source, write_cv_json,
    save_test_results,
)
from src.metrics import agreement_scores, metric_bundle, format_bundle


def load_data(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _make_xgb(args, patience=None, n_estimators_override=None):
    """Create an XGBClassifier from argparse hyperparameters."""
    kwargs = dict(
        n_estimators=n_estimators_override or args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        objective="multi:softprob",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        random_state=args.seed,
        tree_method="hist",
        n_jobs=1,
        verbosity=0,
    )
    if patience is not None:
        kwargs["early_stopping_rounds"] = patience
    return XGBClassifier(**kwargs)


# ── LOSO Cross-Validation ────────────────────────────────────────────────────

def run_loso_cv(features, labels, datasets, feature_names, task_map, args):
    folds = loso_splits(datasets, TRAIN_DATASETS)

    print(f"\n{'='*70}")
    print(f"  LOSO Cross-Validation ({len(folds)} folds) | XGBoost")
    print(f"{'='*70}")

    fold_results = []
    agg_true_list, agg_pred_list = [], []
    importances_accum = np.zeros(features.shape[1])

    for train_idx, val_idx, held_out in folds:
        print(f"\n  -- Fold: hold out '{held_out}' --")
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        train_sources = np.unique(datasets[train_idx])
        print(f"    Train: {len(train_idx)} seqs from {list(train_sources)}  "
              f"Val: {len(val_idx)} seqs from [{held_out}]")
        print(f"    Val label dist: {dict(Counter(y_val))}")

        if args.normalize == "per-source":
            X_train = standardize_per_source(X_train, datasets[train_idx])
            X_val = standardize_per_source(X_val, datasets[val_idx])
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        model = _make_xgb(args, patience=args.patience)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        best_n = model.best_iteration + 1
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1_m = f1_score(y_val, y_pred, average="macro")
        f1_w = f1_score(y_val, y_pred, average="weighted")
        agr = agreement_scores(y_val, y_pred)

        print(f"    acc={acc:.4f}  f1_macro={f1_m:.4f}  f1_weighted={f1_w:.4f}  "
              f"mcc={agr['mcc']:.4f}  kappa={agr['kappa']:.4f}  "
              f"best_n_estimators={best_n}")

        fold_results.append({
            "held_out_source": held_out,
            "accuracy": float(acc),
            "f1_macro": float(f1_m),
            "f1_weighted": float(f1_w),
            "mcc": agr["mcc"],
            "kappa": agr["kappa"],
            "best_n_estimators": best_n,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "confusion_matrix": confusion_matrix(
                y_val, y_pred, labels=list(range(N_CLASSES))).tolist(),
        })
        agg_true_list.append(y_val)
        agg_pred_list.append(y_pred)
        importances_accum += model.feature_importances_

    print_loso_summary(fold_results,
                       ["accuracy", "f1_macro", "f1_weighted", "mcc", "kappa"])

    agg_true = np.concatenate(agg_true_list)
    agg_pred = np.concatenate(agg_pred_list)
    bundle = metric_bundle(agg_true, agg_pred, n_classes=N_CLASSES, n_boot=2000)
    print(f"\n  Chance-corrected agreement (aggregated, 95% bootstrap CI):")
    print(f"    {format_bundle(bundle, keys=['mcc', 'kappa', 'balanced_accuracy'])}")

    target_names = [TASK_LABEL_MAP[i] for i in range(N_CLASSES)]
    print(f"\n  Classification Report (all folds aggregated):")
    print(classification_report(agg_true, agg_pred, target_names=target_names, digits=4))

    cm = confusion_matrix(agg_true, agg_pred)
    print("  Confusion Matrix:")
    print(f"  {'':>10s}  " + "  ".join(f"{s:>7s}" for s in TASK_SHORT))
    for i, row in enumerate(cm):
        print(f"  {TASK_SHORT[i]:>10s}  " + "  ".join(f"{v:7d}" for v in row))

    avg_importance = importances_accum / len(folds)
    sorted_idx = np.argsort(avg_importance)[::-1]
    print(f"\n  Top 15 Features (avg importance across folds):")
    for rank, idx in enumerate(sorted_idx[:15]):
        print(f"    {rank+1:2d}. {feature_names[idx]:30s}  {avg_importance[idx]:.4f}")

    if args.save_model:
        save_dir = Path(args.save_model)
        save_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        cv_path = save_dir / f"xgboost_{args.label_scheme}_loso_cv_{date_str}.pkl"
        save_cv_checkpoint(cv_path, fold_results, model_config={
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
        }, extra={
            "feature_names": feature_names,
            "task_label_map": task_map,
        })

    return fold_results


# ── Full Training + Test Evaluation ──────────────────────────────────────────

def run_full_training(features, labels, train_sources, n_estimators, test_data,
                      feature_names, task_map, args):
    print(f"\n{'='*70}")
    print(f"  Full Training | XGBoost | n_estimators={n_estimators}")
    print(f"{'='*70}")

    scaler = None
    if args.normalize == "per-source":
        X_train = standardize_per_source(features, train_sources)
    else:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features)

    test_eval_set = []
    X_test, y_test = None, None
    if test_data is not None:
        if args.normalize == "per-source":
            test_sources = np.array([m["dataset"] for m in test_data["metadata"]])
            X_test = standardize_per_source(test_data["features"], test_sources)
        else:
            X_test = scaler.transform(test_data["features"])
        y_test = test_data["labels"]
        test_eval_set = [(X_test, y_test)]
        print(f"  Train: {len(features)} seqs | Test: {len(y_test)} seqs")

    model = _make_xgb(args, patience=None, n_estimators_override=n_estimators)
    model.fit(X_train, labels, eval_set=test_eval_set, verbose=20)

    # Per-round test monitoring
    if test_data is not None and args.save_model:
        evals = model.evals_result()
        if evals and "validation_0" in evals:
            test_log = []
            metrics_dict = evals["validation_0"]
            n_rounds = len(next(iter(metrics_dict.values())))
            for i in range(n_rounds):
                entry = {"round": i + 1}
                for metric_name, values in metrics_dict.items():
                    entry[metric_name] = float(values[i])
                test_log.append(entry)
            save_test_epoch_log(test_log, f"xgboost_{args.label_scheme}", args.save_model)

    # Final test evaluation
    if test_data is not None:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_m = f1_score(y_test, y_pred, average="macro")
        f1_w = f1_score(y_test, y_pred, average="weighted")
        per_class = f1_score(y_test, y_pred, average=None, labels=list(range(N_CLASSES)))

        bundle = metric_bundle(y_test, y_pred, n_classes=N_CLASSES, n_boot=2000)
        print(f"\n  Test Results:")
        print(f"    Accuracy:     {acc:.4f}")
        print(f"    F1 (macro):   {f1_m:.4f}")
        print(f"    F1 (weight):  {f1_w:.4f}")
        print(f"    Agreement (95% bootstrap CI): "
              f"{format_bundle(bundle, keys=['mcc', 'kappa', 'balanced_accuracy'])}")

        target_names = [TASK_LABEL_MAP[i] for i in range(N_CLASSES)]
        print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

        cm = confusion_matrix(y_test, y_pred, labels=list(range(N_CLASSES)))
        print("  Confusion Matrix:")
        print(f"  {'':>10s}  " + "  ".join(f"{s:>7s}" for s in TASK_SHORT))
        for i, row in enumerate(cm):
            print(f"  {TASK_SHORT[i]:>10s}  " + "  ".join(f"{v:7d}" for v in row))

        if args.save_model:
            save_test_results(args.save_model, f"xgboost_{args.label_scheme}", {
                "model": "xgboost", "label_scheme": args.label_scheme,
                "normalize": args.normalize,
                "accuracy": float(acc), "f1_macro": float(f1_m), "f1_weighted": float(f1_w),
                "mcc": bundle["mcc"]["value"], "kappa": bundle["kappa"]["value"],
                "confusion_matrix": cm.tolist(),
                "labels": [task_map[i] for i in range(N_CLASSES)],
            })

    # Save model
    if args.save_model:
        save_dir = Path(args.save_model)
        save_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        save_path = save_dir / f"stage1_xgboost_{args.label_scheme}_{date_str}.pkl"

        save_obj = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "task_label_map": task_map,
            "model_config": {
                "n_estimators": n_estimators,
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
            },
            "normalize": args.normalize,
            "label_scheme": args.label_scheme,
            "mode": "full",
        }
        with open(save_path, "wb") as f:
            pickle.dump(save_obj, f)
        print(f"  Saved model to {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1: XGBoost Training")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to train_summary.pkl")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Path to test_summary.pkl (required for full mode)")
    parser.add_argument("--mode", choices=["cv", "full", "both"], default="cv",
                        help="cv=LOSO only, full=refit+test, both=CV then full")
    parser.add_argument("--save-model", type=str, default=None,
                        help="Directory to save checkpoints")
    parser.add_argument("--cv-checkpoint", type=str, default=None,
                        help="Path to prior LOSO CV checkpoint (for full mode)")
    parser.add_argument("--fixed-iterations", type=int, default=None,
                        help="Override n_estimators for full mode")
    parser.add_argument("--n-estimators", type=int, default=1000,
                        help="Max boosting rounds (upper bound for CV early stopping)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (rounds)")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=int, default=3)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-scheme", choices=["fine", "coarse"], default="fine",
                        help="fine=7 tasks, coarse=3 interaction groups")
    parser.add_argument("--normalize", choices=["global", "per-source"], default="global",
                        help="global=one scaler on pooled train; per-source=each "
                             "source standardized by its own stats (Option A)")
    parser.add_argument("--drop-group", type=str, default=None,
                        help="Ablation: drop one summary feature group (see "
                             "SUMMARY_FEATURE_GROUPS in constants)")
    parser.add_argument("--keep-only", type=str, default=None,
                        help="Ablation: keep only one summary feature group")
    parser.add_argument("--cv-json", type=str, default=None,
                        help="Write machine-readable LOSO CV summary to this path")
    args = parser.parse_args()

    if args.mode in ("full", "both") and args.test_data is None:
        parser.error("--test-data is required for full/both mode")
    if args.mode == "full" and args.cv_checkpoint is None and args.fixed_iterations is None:
        parser.error("--cv-checkpoint or --fixed-iterations required for full mode")

    np.random.seed(args.seed)

    # Activate the label scheme: rebind module globals so all helpers below
    # (which read N_CLASSES / TASK_LABEL_MAP / TASK_SHORT) use the active scheme.
    global N_CLASSES, TASK_LABEL_MAP, TASK_SHORT
    scheme = get_label_scheme(args.label_scheme)
    N_CLASSES = C.N_CLASSES = scheme["n_classes"]
    TASK_LABEL_MAP = C.TASK_LABEL_MAP = scheme["label_map"]
    TASK_SHORT = C.TASK_SHORT = scheme["short"]
    print(f"Label scheme: {args.label_scheme} ({N_CLASSES} classes)")

    print("Loading training data...")
    data = load_data(args.data)
    features = data["features"]
    labels = scheme["remap"](data["labels"])
    metadata = data["metadata"]
    feature_names = data["feature_names"]
    task_map = scheme["label_map"]

    datasets = np.array([m["dataset"] for m in metadata])

    # Feature-group ablation: mask columns before any training. Groups are
    # resolved against the canonical SUMMARY_FEATURE_NAMES order (the pkl may
    # store generic column labels), with a dimension guard for safety.
    keep_idx = None
    if args.drop_group or args.keep_only:
        canon = C.SUMMARY_FEATURE_NAMES
        if features.shape[1] != len(canon):
            raise ValueError(
                f"summary feature dim {features.shape[1]} != {len(canon)} canonical; "
                "cannot map feature groups"
            )
        grp = args.drop_group or args.keep_only
        sel = resolve_group_features(SUMMARY_FEATURE_GROUPS, grp)
        if args.drop_group:
            keep_idx = select_feature_indices(canon, drop=sel)
            print(f"  Ablation: dropping group '{grp}' ({len(sel)} features)")
        else:
            keep_idx = select_feature_indices(canon, keep=sel)
            print(f"  Ablation: keeping only group '{grp}' ({len(sel)} features)")
        features = features[:, keep_idx]
        feature_names = [canon[i] for i in keep_idx]

    print(f"  {len(features)} sequences, {features.shape[1]} features")
    print(f"  Sources: {dict(Counter(datasets))}")

    fold_results = None

    if args.mode in ("cv", "both"):
        fold_results = run_loso_cv(
            features, labels, datasets, feature_names, task_map, args,
        )
        if args.cv_json:
            write_cv_json(
                args.cv_json, fold_results, model="xgboost",
                label_scheme=args.label_scheme, normalize=args.normalize,
                seed=args.seed,
                ablation={"drop_group": args.drop_group, "keep_only": args.keep_only},
            )

    if args.mode in ("full", "both"):
        n_estimators = resolve_n_epochs(args, fold_results)
        print(f"\n  Using n_estimators={n_estimators} for full training")

        test_data = load_data(args.test_data) if args.test_data else None
        if test_data is not None:
            test_data["labels"] = scheme["remap"](test_data["labels"])
            if keep_idx is not None:
                test_data["features"] = test_data["features"][:, keep_idx]
        run_full_training(
            features, labels, datasets, n_estimators, test_data, feature_names,
            task_map, args,
        )


if __name__ == "__main__":
    main()
