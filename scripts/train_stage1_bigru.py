"""
Stage 1: BiGRU on Temporal Features
====================================
Sequence-level classification using per-timestep feature trajectories.

Modes:
    cv    — LOSO cross-validation only (default)
    full  — Refit on all training data, evaluate on test
    both  — CV then full (uses median best_epoch from CV)

Usage:
    # LOSO cross-validation
    python train_stage1_bigru.py --data train_temporal.pkl --save-model checkpoints/

    # Full training from CV checkpoint
    python train_stage1_bigru.py --data train_temporal.pkl --mode full \
        --test-data test_temporal.pkl --cv-checkpoint checkpoints/bigru_loso_cv.pkl \
        --save-model checkpoints/

    # Both in one run
    python train_stage1_bigru.py --data train_temporal.pkl --mode both \
        --test-data test_temporal.pkl --save-model checkpoints/
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

import src.constants as C
from src.constants import N_CLASSES, TASK_SHORT, TASK_LABEL_MAP, TRAIN_DATASETS
from src.constants import (
    get_label_scheme, select_feature_indices, resolve_group_features,
    TEMPORAL_FEATURE_GROUPS,
)
from src.models.bigru import BiGRUClassifier, BiGRUAttentionClassifier
from src.datasets import (
    HoverSequenceDataset, AugmentableDataset, collate_sequences as collate_fn,
    compute_feature_stats, normalize_sequences, normalize_sequences_per_source,
)
from src.training import (
    loso_splits, print_loso_summary, save_cv_checkpoint,
    save_test_epoch_log, resolve_n_epochs, write_cv_json, save_test_results,
)
from src.metrics import agreement_scores, metric_bundle, format_bundle


# ── Training / Evaluation helpers ────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for x, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_true = [], []
    for x, y, lengths in loader:
        x, lengths = x.to(device), lengths.to(device)
        logits = model(x, lengths)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_true.append(y.numpy())
    return np.concatenate(all_preds), np.concatenate(all_true)


# ── LOSO Cross-Validation ────────────────────────────────────────────────────

def run_loso_cv(sequences, labels, datasets, input_dim, feature_names, task_map,
                args, device, pin_memory, num_workers):
    folds = loso_splits(datasets, TRAIN_DATASETS)

    model_variant = "BiGRU+Attn" if args.attention else "BiGRU"
    print(f"\n{'='*70}")
    print(f"  LOSO Cross-Validation ({len(folds)} folds) | {model_variant}")
    print(f"{'='*70}")

    def make_model():
        cls = BiGRUAttentionClassifier if args.attention else BiGRUClassifier
        return cls(
            input_dim=input_dim, hidden_dim=args.hidden_dim,
            num_classes=N_CLASSES, dropout=args.dropout,
        ).to(device)

    fold_results = []
    agg_true_list, agg_pred_list = [], []

    for train_idx, val_idx, held_out in folds:
        print(f"\n  -- Fold: hold out '{held_out}' --")

        train_seqs = [sequences[i] for i in train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]

        train_sources = np.unique(datasets[train_idx])
        print(f"    Train: {len(train_idx)} seqs from {list(train_sources)}  "
              f"Val: {len(val_idx)} seqs from [{held_out}]")
        print(f"    Val label dist: {dict(Counter(val_labels))}")

        if args.normalize == "per-source":
            train_seqs_norm = normalize_sequences_per_source(train_seqs, datasets[train_idx])
            val_seqs_norm = normalize_sequences_per_source(val_seqs, datasets[val_idx])
        else:
            mean, std = compute_feature_stats(train_seqs)
            train_seqs_norm = normalize_sequences(train_seqs, mean, std)
            val_seqs_norm = normalize_sequences(val_seqs, mean, std)

        if args.augment > 0:
            train_dataset = AugmentableDataset(
                train_seqs_norm, train_labels, augment_min_frac=args.augment,
            )
        else:
            train_dataset = HoverSequenceDataset(train_seqs_norm, train_labels)
        val_dataset = HoverSequenceDataset(val_seqs_norm, val_labels)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
        )

        class_counts = np.bincount(train_labels, minlength=N_CLASSES).astype(np.float32)
        class_counts[class_counts == 0] = 1.0
        class_weights = len(train_labels) / (N_CLASSES * class_counts)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        model = make_model()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        best_f1, best_state, patience_counter, best_epoch = -1, None, 0, 0

        for epoch in range(args.epochs):
            if args.augment > 0:
                train_dataset.train_mode()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            preds, true = evaluate(model, val_loader, device)
            f1_m = f1_score(true, preds, average="macro")
            scheduler.step(f1_m)

            if f1_m > best_f1:
                best_f1 = f1_m
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                best_epoch = epoch + 1
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or patience_counter >= args.patience:
                acc = accuracy_score(true, preds)
                print(f"    Epoch {epoch+1:3d}: loss={train_loss:.4f}  acc={acc:.4f}  "
                      f"f1_macro={f1_m:.4f}  best_f1={best_f1:.4f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.6f}")

            if patience_counter >= args.patience:
                print(f"    Early stop at epoch {epoch+1}")
                break

        model.load_state_dict(best_state)
        preds, true = evaluate(model, val_loader, device)
        acc = accuracy_score(true, preds)
        f1_m = f1_score(true, preds, average="macro")
        f1_w = f1_score(true, preds, average="weighted")
        agr = agreement_scores(true, preds)

        print(f"    Best: acc={acc:.4f}  f1_macro={f1_m:.4f}  f1_weighted={f1_w:.4f}  "
              f"mcc={agr['mcc']:.4f}  kappa={agr['kappa']:.4f}  epoch={best_epoch}")

        fold_results.append({
            "held_out_source": held_out,
            "accuracy": float(acc),
            "f1_macro": float(f1_m),
            "f1_weighted": float(f1_w),
            "mcc": agr["mcc"],
            "kappa": agr["kappa"],
            "best_epoch": best_epoch,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "confusion_matrix": confusion_matrix(
                true, preds, labels=list(range(N_CLASSES))).tolist(),
        })
        agg_true_list.append(true)
        agg_pred_list.append(preds)

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

    if args.save_model:
        save_dir = Path(args.save_model)
        save_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        suffix = "_attn" if args.attention else ""
        cv_path = save_dir / f"bigru_{args.label_scheme}{suffix}_loso_cv_{date_str}.pkl"
        save_cv_checkpoint(cv_path, fold_results, model_config={
            "input_dim": input_dim, "hidden_dim": args.hidden_dim,
            "num_classes": N_CLASSES, "dropout": args.dropout,
            "attention": args.attention,
        }, extra={
            "feature_names": feature_names,
            "task_label_map": task_map,
        })

    return fold_results


# ── Full Training + Test Evaluation ──────────────────────────────────────────

def run_full_training(sequences, labels, train_sources, n_epochs, test_data,
                      input_dim, feature_names, task_map, args, device,
                      pin_memory, num_workers):
    model_variant = "BiGRU+Attn" if args.attention else "BiGRU"
    print(f"\n{'='*70}")
    print(f"  Full Training | {model_variant} | epochs={n_epochs}")
    print(f"{'='*70}")

    # Train on ALL training sources for the LOSO median epoch. That median was
    # chosen on held-out *sources*, so it is the cross-source-aligned stopping
    # rule for the unseen test source. An internal within-source val split is
    # deliberately avoided: it optimizes within-source fit, which diverges from
    # cross-source generalization (the test objective).
    if args.normalize == "per-source":
        mean, std = None, None
        all_seqs_norm = normalize_sequences_per_source(sequences, train_sources)
    else:
        mean, std = compute_feature_stats(sequences)
        all_seqs_norm = normalize_sequences(sequences, mean, std)

    train_dataset = HoverSequenceDataset(all_seqs_norm, labels)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
    )

    test_loader = None
    if test_data is not None:
        if args.normalize == "per-source":
            test_sources = np.array([m["dataset"] for m in test_data["metadata"]])
            test_seqs_norm = normalize_sequences_per_source(test_data["sequences"], test_sources)
        else:
            test_seqs_norm = normalize_sequences(test_data["sequences"], mean, std)
        test_dataset = HoverSequenceDataset(test_seqs_norm, test_data["labels"])
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
        )
        print(f"  Train: {len(labels)} seqs | Test: {len(test_data['labels'])} seqs")

    cls = BiGRUAttentionClassifier if args.attention else BiGRUClassifier
    model = cls(
        input_dim=input_dim, hidden_dim=args.hidden_dim,
        num_classes=N_CLASSES, dropout=args.dropout,
    ).to(device)

    class_counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    class_weights = len(labels) / (N_CLASSES * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    test_epoch_log = []

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        log_line = f"    Epoch {epoch+1:3d}/{n_epochs}: loss={train_loss:.4f}"

        # Test metrics are logged per epoch for inspection only; they never
        # influence training or model selection (no best-restore on test).
        if test_loader is not None:
            preds, true = evaluate(model, test_loader, device)
            test_acc = accuracy_score(true, preds)
            test_f1 = f1_score(true, preds, average="macro")
            test_epoch_log.append({
                "epoch": epoch + 1,
                "test_accuracy": float(test_acc),
                "test_f1_macro": float(test_f1),
            })
            log_line += f"  test_acc={test_acc:.4f}  test_f1={test_f1:.4f}"

        log_line += f"  lr={optimizer.param_groups[0]['lr']:.6f}"
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == n_epochs - 1:
            print(log_line, flush=True)

    # Save per-epoch test log
    if test_epoch_log and args.save_model:
        suffix = "_attn" if args.attention else ""
        save_test_epoch_log(test_epoch_log, f"bigru_{args.label_scheme}{suffix}", args.save_model)

    # Final test evaluation
    if test_loader is not None:
        preds, true = evaluate(model, test_loader, device)
        acc = accuracy_score(true, preds)
        f1_m = f1_score(true, preds, average="macro")
        f1_w = f1_score(true, preds, average="weighted")

        bundle = metric_bundle(true, preds, n_classes=N_CLASSES, n_boot=2000)
        print(f"\n  Final Test Results:")
        print(f"    Accuracy:     {acc:.4f}")
        print(f"    F1 (macro):   {f1_m:.4f}")
        print(f"    F1 (weight):  {f1_w:.4f}")
        print(f"    Agreement (95% bootstrap CI): "
              f"{format_bundle(bundle, keys=['mcc', 'kappa', 'balanced_accuracy'])}")

        target_names = [TASK_LABEL_MAP[i] for i in range(N_CLASSES)]
        print(classification_report(true, preds, target_names=target_names, digits=4))

        cm = confusion_matrix(true, preds, labels=list(range(N_CLASSES)))
        print("  Confusion Matrix:")
        print(f"  {'':>10s}  " + "  ".join(f"{s:>7s}" for s in TASK_SHORT))
        for i, row in enumerate(cm):
            print(f"  {TASK_SHORT[i]:>10s}  " + "  ".join(f"{v:7d}" for v in row))

        if args.save_model:
            suffix = "_attn" if args.attention else ""
            save_test_results(args.save_model, f"bigru{suffix}_{args.label_scheme}", {
                "model": model_variant, "label_scheme": args.label_scheme,
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
        suffix = "_attn" if args.attention else ""
        save_path = save_dir / f"stage1_bigru_{args.label_scheme}{suffix}_{date_str}.pt"

        save_obj = {
            "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
            "feature_mean": mean,
            "feature_std": std,
            "feature_names": feature_names,
            "task_label_map": task_map,
            "model_variant": model_variant,
            "normalize": args.normalize,
            "label_scheme": args.label_scheme,
            "model_config": {
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "num_classes": N_CLASSES,
                "dropout": args.dropout,
                "attention": args.attention,
            },
            "mode": "full",
            "n_epochs": n_epochs,
        }
        torch.save(save_obj, save_path)
        print(f"  Saved model to {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1: BiGRU Training")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to train_temporal.pkl")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Path to test_temporal.pkl (required for full mode)")
    parser.add_argument("--mode", choices=["cv", "full", "both"], default="cv",
                        help="cv=LOSO only, full=refit+test, both=CV then full")
    parser.add_argument("--save-model", type=str, default=None,
                        help="Directory to save checkpoints")
    parser.add_argument("--cv-checkpoint", type=str, default=None,
                        help="Path to prior LOSO CV checkpoint (for full mode)")
    parser.add_argument("--fixed-epochs", type=int, default=None,
                        help="Override epoch count for full mode")

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs per fold (CV mode)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--label-scheme", choices=["fine", "coarse"], default="fine",
                        help="fine=7 tasks, coarse=3 interaction groups")
    parser.add_argument("--normalize", choices=["global", "per-source"], default="global",
                        help="global=pooled-train stats; per-source=each source "
                             "standardized by its own stats (Option A)")
    parser.add_argument("--attention", action="store_true",
                        help="Use attention pooling instead of final hidden")
    parser.add_argument("--augment", type=float, default=0.0, metavar="FRAC",
                        help="Subsequence augmentation min fraction (e.g. 0.7)")
    parser.add_argument("--drop-group", type=str, default=None,
                        help="Ablation: drop one temporal feature group "
                             "(stateless/stateful, see TEMPORAL_FEATURE_GROUPS)")
    parser.add_argument("--keep-only", type=str, default=None,
                        help="Ablation: keep only one temporal feature group")
    parser.add_argument("--shuffle-time", action="store_true",
                        help="H4 ablation: randomly permute timestep order within "
                             "each sequence (destroys temporal order, keeps features)")
    parser.add_argument("--cv-json", type=str, default=None,
                        help="Write machine-readable LOSO CV summary to this path")
    args = parser.parse_args()

    if args.mode in ("full", "both") and args.test_data is None:
        parser.error("--test-data is required for full/both mode")
    if args.mode == "full" and args.cv_checkpoint is None and args.fixed_epochs is None:
        parser.error("--cv-checkpoint or --fixed-epochs required for full mode")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Activate the label scheme: rebind module globals so all helpers below
    # (which read N_CLASSES / TASK_LABEL_MAP / TASK_SHORT) use the active scheme.
    global N_CLASSES, TASK_LABEL_MAP, TASK_SHORT
    scheme = get_label_scheme(args.label_scheme)
    N_CLASSES = C.N_CLASSES = scheme["n_classes"]
    TASK_LABEL_MAP = C.TASK_LABEL_MAP = scheme["label_map"]
    TASK_SHORT = C.TASK_SHORT = scheme["short"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    pin_memory = device.type == "cuda"
    num_workers = 0 if device.type in ("mps", "cpu") else 4

    model_variant = "BiGRU+Attn" if args.attention else "BiGRU"
    print(f"Device: {device}  |  Model: {model_variant}  |  hidden_dim={args.hidden_dim}"
          f"  |  Label scheme: {args.label_scheme} ({N_CLASSES} classes)")

    print("Loading training data...")
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    sequences = data["sequences"]
    labels = scheme["remap"](data["labels"])
    metadata = data["metadata"]
    feature_names = data["feature_names"]
    task_map = scheme["label_map"]

    datasets = np.array([m["dataset"] for m in metadata])

    # Feature-group ablation: mask per-timestep feature dimensions. Groups are
    # resolved against the canonical TEMPORAL_FEATURE_NAMES order (the pkl stores
    # generic temporal_N labels), with a dimension guard for safety.
    keep_idx = None
    if args.drop_group or args.keep_only:
        canon = C.TEMPORAL_FEATURE_NAMES
        if sequences[0].shape[1] != len(canon):
            raise ValueError(
                f"temporal dim {sequences[0].shape[1]} != {len(canon)} canonical; "
                "cannot map feature groups"
            )
        grp = args.drop_group or args.keep_only
        sel = resolve_group_features(TEMPORAL_FEATURE_GROUPS, grp)
        if args.drop_group:
            keep_idx = select_feature_indices(canon, drop=sel)
            print(f"  Ablation: dropping group '{grp}' ({len(sel)} features)")
        else:
            keep_idx = select_feature_indices(canon, keep=sel)
            print(f"  Ablation: keeping only group '{grp}' ({len(sel)} features)")
        sequences = [s[:, keep_idx] for s in sequences]
        feature_names = [canon[i] for i in keep_idx]

    # H4 ablation: destroy temporal order within each sequence (keep features).
    if args.shuffle_time:
        sequences = [s[np.random.permutation(s.shape[0])] for s in sequences]
        print("  Ablation: timestep order shuffled within each sequence (H4)")

    input_dim = sequences[0].shape[1]
    print(f"  {len(sequences)} sequences, dim={input_dim}")
    print(f"  Sources: {dict(Counter(datasets))}")

    fold_results = None

    if args.mode in ("cv", "both"):
        fold_results = run_loso_cv(
            sequences, labels, datasets, input_dim, feature_names, task_map,
            args, device, pin_memory, num_workers,
        )
        if args.cv_json:
            write_cv_json(
                args.cv_json, fold_results,
                model="bigru_attn" if args.attention else "bigru",
                label_scheme=args.label_scheme, normalize=args.normalize,
                seed=args.seed,
                ablation={
                    "drop_group": args.drop_group, "keep_only": args.keep_only,
                    "shuffle_time": args.shuffle_time,
                },
            )

    if args.mode in ("full", "both"):
        n_epochs = resolve_n_epochs(args, fold_results)
        print(f"\n  Using n_epochs={n_epochs} for full training")

        test_data = None
        if args.test_data:
            with open(args.test_data, "rb") as f:
                test_data = pickle.load(f)
            test_data["labels"] = scheme["remap"](test_data["labels"])
            if keep_idx is not None:
                test_data["sequences"] = [s[:, keep_idx] for s in test_data["sequences"]]
            if args.shuffle_time:
                test_data["sequences"] = [
                    s[np.random.permutation(s.shape[0])] for s in test_data["sequences"]
                ]

        run_full_training(
            sequences, labels, datasets, n_epochs, test_data, input_dim,
            feature_names, task_map, args, device, pin_memory, num_workers,
        )


if __name__ == "__main__":
    main()
