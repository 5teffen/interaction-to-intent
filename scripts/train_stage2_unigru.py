"""
Stage 2: UniGRU + CRF for Per-Timestep Task Prediction
=======================================================
Predicts the active atomic task at each timestep of a composed exploration session.

Supports Variant A (24-dim, raw concatenation — per-source pre-composed)
and Variant B (15-dim, feature stitching — composed on-the-fly from atomics).

Modes:
    cv    — LOSO cross-validation only (default)
    full  — Refit on all training data, evaluate on test
    both  — CV then full (uses median best_epoch from CV)

Usage:
    # LOSO CV on Variant A
    python train_stage2_unigru.py --variant A --data-dir data/processed/ \
        --save-model checkpoints/

    # Full training from CV checkpoint
    python train_stage2_unigru.py --variant A --data-dir data/processed/ \
        --mode full --test-data test_stage2_variantA.pkl \
        --cv-checkpoint checkpoints/stage2_A_loso_cv.pkl \
        --save-model checkpoints/

    # Both in one run
    python train_stage2_unigru.py --variant B --data-dir data/processed/ \
        --mode both --test-data test_stage2_variantB.pkl \
        --save-model checkpoints/
"""

import os
# Let unsupported CRF ops fall back to CPU instead of crashing an overnight run.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
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
from src.constants import get_label_scheme, TEMPORAL_FEATURE_NAMES, CUMULATIVE_FEATURE_NAMES
from src.models.stage2 import UniGRU_CRF
from src.datasets import (
    TimestepLabelDataset, collate_timesteps as collate_fn,
    compute_feature_stats, normalize_sequences,
    normalize_sequences_per_source, normalize_sequences_per_session,
)
from src.metrics import (
    segment_f1 as compute_segment_f1, boundary_f1 as compute_boundary_f1,
    agreement_scores, metric_bundle, format_bundle,
)
from src.compose import stitch_feature_matrices, generate_sessions_any
from src.training import (
    print_loso_summary, save_cv_checkpoint,
    save_test_epoch_log, resolve_n_epochs, write_cv_json, save_test_results,
    decay_cumulative,
)


def _cumulative_indices():
    """Canonical column indices of the monotonic cumulative temporal features."""
    return [TEMPORAL_FEATURE_NAMES.index(n) for n in CUMULATIVE_FEATURE_NAMES]


class _Tee:
    """Duplicate stdout to a log file for overnight runs."""
    def __init__(self, stream, path):
        self.stream = stream
        self.log = open(path, "a", buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.log.write(data)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def close(self):
        self.log.close()

    def __del__(self):
        try:
            self.log.close()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _normalize_split(seqs, sources, args, variant):
    """Apply the chosen normalization to a list of sessions.

    Returns (normalized_seqs, mean, std). mean/std are None unless global.
    Per-source applies for Variant A (one source per session); Variant B
    sessions mix sources, so per-source falls back to per-session.
    """
    if args.normalize == "per-source":
        if variant == "A" and sources is not None:
            return normalize_sequences_per_source(seqs, np.asarray(sources)), None, None
        return normalize_sequences_per_session(seqs), None, None
    mean, std = compute_feature_stats(seqs)
    return normalize_sequences(seqs, mean, std), mean, std


def _save_cv_progress(args, variant, fold_results, input_dim, feature_names, task_map):
    """Write the LOSO CV checkpoint with current fold_results (called per fold)."""
    save_dir = Path(args.save_model)
    save_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    cv_path = save_dir / f"stage2_{variant}_{args.label_scheme}_loso_cv_{date_str}.pkl"
    save_cv_checkpoint(cv_path, fold_results, model_config={
        "input_dim": input_dim, "hidden_dim": args.hidden_dim,
        "num_classes": N_CLASSES, "dropout": args.dropout,
    }, extra={
        "variant": variant, "feature_names": feature_names,
        "task_label_map": task_map, "label_scheme": args.label_scheme,
        "normalize": args.normalize,
    })


def _save_full_model(args, variant, model, mean, std, feature_names, task_map,
                     input_dim, n_epochs, filename):
    """Write a full-training model checkpoint (used for periodic + final saves)."""
    save_dir = Path(args.save_model)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_obj = {
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "feature_mean": mean, "feature_std": std,
        "feature_names": feature_names, "task_label_map": task_map,
        "variant": variant, "model_type": "UniGRU+CRF",
        "label_scheme": args.label_scheme, "normalize": args.normalize,
        "model_config": {
            "input_dim": input_dim, "hidden_dim": args.hidden_dim,
            "num_classes": N_CLASSES, "dropout": args.dropout,
        },
        "mode": "full", "n_epochs": n_epochs,
    }
    path = save_dir / filename
    torch.save(save_obj, path)
    return path


# ── Training / Evaluation helpers ────────────────────────────────────────────

def train_crf_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for x, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
        optimizer.zero_grad()
        loss = model(x, y, lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_crf(model, loader, device):
    model.eval()
    all_preds, all_true = [], []
    for x, y, lengths in loader:
        x, lengths = x.to(device), lengths.to(device)
        decoded = model.decode(x, lengths)
        for i, length in enumerate(lengths):
            l = length.item()
            all_preds.append(np.array(decoded[i][:l]))
            all_true.append(y[i][:l].numpy())
    return all_preds, all_true


def _compute_all_metrics(preds_list, true_list, boundary_tolerance=3):
    """Compute all Stage 2 metrics from per-session prediction lists."""
    preds_flat = np.concatenate(preds_list)
    true_flat = np.concatenate(true_list)
    ts_acc = accuracy_score(true_flat, preds_flat)
    ts_f1_m = f1_score(true_flat, preds_flat, average="macro")
    ts_f1_w = f1_score(true_flat, preds_flat, average="weighted")
    seg_f1 = compute_segment_f1(true_list, preds_list, n_classes=N_CLASSES)
    bnd = compute_boundary_f1(true_list, preds_list, tolerance=boundary_tolerance)
    agr = agreement_scores(true_flat, preds_flat)
    return {
        "ts_accuracy": ts_acc, "ts_f1_macro": ts_f1_m, "ts_f1_weighted": ts_f1_w,
        "segment_f1": seg_f1, "boundary_f1": bnd["f1"],
        "boundary_precision": bnd["precision"], "boundary_recall": bnd["recall"],
        "mcc": agr["mcc"], "kappa": agr["kappa"],
    }


# ── Data loading ─────────────────────────────────────────────────────────────

def load_variant_a_per_source(data_dir: Path) -> dict[str, dict]:
    """Load per-source Variant A pkl files. Returns {source: data_dict}."""
    source_data = {}
    for source in TRAIN_DATASETS:
        path = data_dir / f"stage2_varA_{source}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                source_data[source] = pickle.load(f)
    return source_data


def load_variant_b_atomics(data_dir: Path) -> dict:
    """Load Variant B atomics pkl."""
    path = data_dir / "stage2_varB_atomics.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def compose_sessions_from_atomics(
    atomic_seqs, atomic_labels, n_sessions, min_tasks, max_tasks, rng,
):
    """Compose Variant B sessions from atomic feature matrices."""
    session_indices = generate_sessions_any(
        len(atomic_seqs), n_sessions, (min_tasks, max_tasks), rng,
    )
    sequences, labels = [], []
    for indices in session_indices:
        matrices = [atomic_seqs[idx] for idx in indices]
        task_types = [atomic_labels[idx] for idx in indices]
        stitched = stitch_feature_matrices(matrices)
        ts_labels = np.concatenate(
            [[t] * m.shape[0] for m, t in zip(matrices, task_types)]
        ).astype(np.int64)
        sequences.append(stitched)
        labels.append(ts_labels)
    return sequences, labels


# ── LOSO Cross-Validation ────────────────────────────────────────────────────

def run_loso_cv(variant, data_dir, args, scheme, device, pin_memory, num_workers):
    model_name = "UniGRU+CRF"
    print(f"\n{'='*70}")
    print(f"  LOSO Cross-Validation (3 folds) | {model_name} | Variant {variant}")
    print(f"{'='*70}")

    task_map = scheme["label_map"]
    if variant == "A":
        source_data = load_variant_a_per_source(data_dir)
        if not source_data:
            raise FileNotFoundError(f"No per-source Variant A files in {data_dir}")
        input_dim = next(iter(source_data.values()))["sequences"][0].shape[1]
        feature_names = next(iter(source_data.values()))["feature_names"]
        # Remap per-timestep label arrays to the active scheme.
        for src in source_data:
            source_data[src]["labels"] = [
                scheme["remap"](l) for l in source_data[src]["labels"]
            ]
        # Decay cumulative features so they stay task-local within a session.
        if args.decay < 1.0:
            cum_idx = _cumulative_indices()
            for src in source_data:
                source_data[src]["sequences"] = decay_cumulative(
                    source_data[src]["sequences"], cum_idx, args.decay)
    else:
        atomics = load_variant_b_atomics(data_dir)
        input_dim = atomics["sequences"][0].shape[1]
        feature_names = atomics["feature_names"]
        # Remap atomic (scalar) labels before composing sessions.
        atomics["labels"] = scheme["remap"](atomics["labels"])
        atom_datasets = np.array([m["dataset"] for m in atomics["metadata"]])

    fold_results = []
    agg_preds_flat, agg_true_flat = [], []

    for held_out in TRAIN_DATASETS:
        print(f"\n  -- Fold: hold out '{held_out}' --")
        train_sources = [s for s in TRAIN_DATASETS if s != held_out]

        train_src_arr, val_src_arr = None, None
        if variant == "A":
            train_seqs = []
            train_labels = []
            train_src = []
            for src in train_sources:
                if src in source_data:
                    n = len(source_data[src]["sequences"])
                    train_seqs.extend(source_data[src]["sequences"])
                    train_labels.extend(source_data[src]["labels"])
                    train_src.extend([src] * n)
            val_seqs = source_data[held_out]["sequences"]
            val_labels = source_data[held_out]["labels"]
            train_src_arr = np.array(train_src)
            val_src_arr = np.array([held_out] * len(val_seqs))
        else:
            rng = np.random.default_rng(args.seed + TRAIN_DATASETS.index(held_out))
            train_mask = np.isin(atom_datasets, train_sources)
            val_mask = atom_datasets == held_out

            train_atom_seqs = [atomics["sequences"][i] for i in np.where(train_mask)[0]]
            train_atom_labels = [atomics["labels"][i] for i in np.where(train_mask)[0]]
            val_atom_seqs = [atomics["sequences"][i] for i in np.where(val_mask)[0]]
            val_atom_labels = [atomics["labels"][i] for i in np.where(val_mask)[0]]

            # Match Variant A's session counts: A composes args.n_sessions PER
            # source, so a 2-source training fold has 2 x n_sessions sessions and
            # the held-out val source has 1 x n_sessions.
            n_train_sessions = args.n_sessions * len(train_sources)
            train_seqs, train_labels = compose_sessions_from_atomics(
                train_atom_seqs, train_atom_labels,
                n_train_sessions, args.min_tasks, args.max_tasks, rng,
            )
            val_seqs, val_labels = compose_sessions_from_atomics(
                val_atom_seqs, val_atom_labels,
                args.n_sessions, args.min_tasks, args.max_tasks, rng,
            )

        print(f"    Train: {len(train_seqs)} sessions from {train_sources}  "
              f"Val: {len(val_seqs)} sessions from [{held_out}]", flush=True)

        train_seqs_norm, _, _ = _normalize_split(train_seqs, train_src_arr, args, variant)
        val_seqs_norm, _, _ = _normalize_split(val_seqs, val_src_arr, args, variant)

        train_dataset = TimestepLabelDataset(train_seqs_norm, train_labels)
        val_dataset = TimestepLabelDataset(val_seqs_norm, val_labels)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
        )

        model = UniGRU_CRF(
            input_dim=input_dim, hidden_dim=args.hidden_dim,
            num_classes=N_CLASSES, dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5,
        )

        best_metric, best_state, patience_counter, best_epoch = -1, None, 0, 0

        for epoch in range(args.epochs):
            train_loss = train_crf_one_epoch(model, train_loader, optimizer, device)
            preds_list, true_list = evaluate_crf(model, val_loader, device)
            preds_flat = np.concatenate(preds_list)
            true_flat = np.concatenate(true_list)
            f1_m = f1_score(true_flat, preds_flat, average="macro")
            scheduler.step(f1_m)

            if f1_m > best_metric:
                best_metric = f1_m
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                best_epoch = epoch + 1
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or patience_counter >= args.patience:
                acc = accuracy_score(true_flat, preds_flat)
                seg_f1 = compute_segment_f1(true_list, preds_list, n_classes=N_CLASSES)
                print(f"    Epoch {epoch+1:3d}: loss={train_loss:.4f}  ts_acc={acc:.4f}  "
                      f"ts_f1={f1_m:.4f}  seg_f1={seg_f1:.4f}  best={best_metric:.4f}",
                      flush=True)

            if patience_counter >= args.patience:
                print(f"    Early stop at epoch {epoch+1}", flush=True)
                break

        model.load_state_dict(best_state)
        preds_list, true_list = evaluate_crf(model, val_loader, device)
        metrics = _compute_all_metrics(preds_list, true_list, args.boundary_tolerance)

        print(f"    Best results (epoch {best_epoch}):")
        print(f"      Timestep acc:  {metrics['ts_accuracy']:.4f}")
        print(f"      Timestep F1:   {metrics['ts_f1_macro']:.4f}")
        print(f"      Segment F1:    {metrics['segment_f1']:.4f}")
        print(f"      Boundary F1:   {metrics['boundary_f1']:.4f}")
        print(f"      MCC / kappa:   {metrics['mcc']:.4f} / {metrics['kappa']:.4f}",
              flush=True)

        fold_results.append({
            "held_out_source": held_out,
            "ts_accuracy": float(metrics["ts_accuracy"]),
            "ts_f1_macro": float(metrics["ts_f1_macro"]),
            "ts_f1_weighted": float(metrics["ts_f1_weighted"]),
            "segment_f1": float(metrics["segment_f1"]),
            "boundary_f1": float(metrics["boundary_f1"]),
            "mcc": float(metrics["mcc"]),
            "kappa": float(metrics["kappa"]),
            "best_epoch": best_epoch,
            "n_train": len(train_seqs),
            "n_val": len(val_seqs),
            "confusion_matrix": confusion_matrix(
                np.concatenate(true_list), np.concatenate(preds_list),
                labels=list(range(N_CLASSES))).tolist(),
        })
        agg_preds_flat.append(np.concatenate(preds_list))
        agg_true_flat.append(np.concatenate(true_list))

        # Crash-safety: persist progress after every fold so an interruption
        # never loses completed folds.
        if args.save_model:
            _save_cv_progress(args, variant, fold_results, input_dim,
                              feature_names, task_map)
            fold_ckpt = (Path(args.save_model) /
                         f"stage2_{variant}_{args.label_scheme}_fold-{held_out}.pt")
            torch.save({"model_state": {k: v.cpu() for k, v in best_state.items()},
                        "held_out_source": held_out, "best_epoch": best_epoch,
                        "model_config": {"input_dim": input_dim,
                                         "hidden_dim": args.hidden_dim,
                                         "num_classes": N_CLASSES,
                                         "dropout": args.dropout}},
                       fold_ckpt)

    print_loso_summary(fold_results, [
        "ts_accuracy", "ts_f1_macro", "segment_f1", "boundary_f1", "mcc", "kappa",
    ])

    all_preds = np.concatenate(agg_preds_flat)
    all_true = np.concatenate(agg_true_flat)
    bundle = metric_bundle(all_true, all_preds, n_classes=N_CLASSES, n_boot=2000)
    print(f"\n  Chance-corrected agreement (timestep, aggregated, 95% CI):")
    print(f"    {format_bundle(bundle, keys=['mcc', 'kappa', 'balanced_accuracy'])}")

    target_names = [task_map[i] for i in range(N_CLASSES)]
    print(f"\n  Per-Timestep Classification Report (all folds):")
    print(classification_report(all_true, all_preds, target_names=target_names, digits=4))

    cm = confusion_matrix(all_true, all_preds)
    print("  Confusion Matrix:")
    print(f"  {'':>10s}  " + "  ".join(f"{s:>7s}" for s in TASK_SHORT))
    for i, row in enumerate(cm):
        print(f"  {TASK_SHORT[i]:>10s}  " + "  ".join(f"{v:7d}" for v in row))

    if args.save_model:
        _save_cv_progress(args, variant, fold_results, input_dim, feature_names, task_map)

    return fold_results, input_dim, feature_names, task_map


# ── Full Training + Test Evaluation ──────────────────────────────────────────

def run_full_training(variant, data_dir, n_epochs, test_data, input_dim,
                      feature_names, task_map, args, scheme, device, pin_memory,
                      num_workers):
    model_name = "UniGRU+CRF"
    print(f"\n{'='*70}")
    print(f"  Full Training | {model_name} | Variant {variant} | epochs={n_epochs}")
    print(f"{'='*70}")

    task_map = scheme["label_map"]
    all_src = None
    if variant == "A":
        source_data = load_variant_a_per_source(data_dir)
        all_seqs, all_labels, all_src = [], [], []
        for src in TRAIN_DATASETS:
            if src in source_data:
                seqs = source_data[src]["sequences"]
                all_seqs.extend(seqs)
                all_labels.extend([scheme["remap"](l) for l in source_data[src]["labels"]])
                all_src.extend([src] * len(seqs))
        all_src = np.array(all_src)
        if args.decay < 1.0:
            all_seqs = decay_cumulative(all_seqs, _cumulative_indices(), args.decay)
        if input_dim is None:
            input_dim = all_seqs[0].shape[1]
            feature_names = source_data[TRAIN_DATASETS[0]]["feature_names"]
    else:
        atomics = load_variant_b_atomics(data_dir)
        if input_dim is None:
            input_dim = atomics["sequences"][0].shape[1]
            feature_names = atomics["feature_names"]
        atomics["labels"] = scheme["remap"](atomics["labels"])
        rng = np.random.default_rng(args.seed + 100)
        # Match Variant A: A composes args.n_sessions per source, so full
        # training over all sources uses n_sessions x #sources sessions.
        n_full_sessions = args.n_sessions * len(TRAIN_DATASETS)
        all_seqs, all_labels = compose_sessions_from_atomics(
            atomics["sequences"], atomics["labels"],
            n_full_sessions, args.min_tasks, args.max_tasks, rng,
        )

    print(f"  Train: {len(all_seqs)} sessions")

    all_seqs_norm, mean, std = _normalize_split(all_seqs, all_src, args, variant)

    train_dataset = TimestepLabelDataset(all_seqs_norm, all_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
    )

    test_loader = None
    if test_data is not None:
        test_labels = [scheme["remap"](l) for l in test_data["labels"]]
        if variant == "A" and args.decay < 1.0:
            test_data["sequences"] = decay_cumulative(
                test_data["sequences"], _cumulative_indices(), args.decay)
        if args.normalize == "per-source":
            if variant == "A" and test_data.get("metadata") is not None \
                    and "dataset" in test_data["metadata"][0]:
                test_src = np.array([m["dataset"] for m in test_data["metadata"]])
                test_seqs_norm = normalize_sequences_per_source(test_data["sequences"], test_src)
            else:
                test_seqs_norm = normalize_sequences_per_session(test_data["sequences"])
        else:
            test_seqs_norm = normalize_sequences(test_data["sequences"], mean, std)
        test_dataset = TimestepLabelDataset(test_seqs_norm, test_labels)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory,
        )
        print(f"  Test: {len(test_data['sequences'])} sessions")

    model = UniGRU_CRF(
        input_dim=input_dim, hidden_dim=args.hidden_dim,
        num_classes=N_CLASSES, dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    test_epoch_log = []

    for epoch in range(n_epochs):
        train_loss = train_crf_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        log_line = f"    Epoch {epoch+1:3d}/{n_epochs}: loss={train_loss:.4f}"

        # Test decode is expensive; only run it every --eval-every epochs
        # (and on the final epoch). It is informational only.
        is_last = epoch == n_epochs - 1
        do_eval = test_loader is not None and (
            (epoch + 1) % args.eval_every == 0 or epoch == 0 or is_last
        )
        if do_eval:
            preds_list, true_list = evaluate_crf(model, test_loader, device)
            metrics = _compute_all_metrics(preds_list, true_list, args.boundary_tolerance)
            test_epoch_log.append({
                "epoch": epoch + 1,
                "ts_accuracy": float(metrics["ts_accuracy"]),
                "ts_f1_macro": float(metrics["ts_f1_macro"]),
                "segment_f1": float(metrics["segment_f1"]),
                "boundary_f1": float(metrics["boundary_f1"]),
            })
            log_line += (f"  ts_acc={metrics['ts_accuracy']:.4f}  "
                         f"ts_f1={metrics['ts_f1_macro']:.4f}  "
                         f"seg_f1={metrics['segment_f1']:.4f}")

        log_line += f"  lr={optimizer.param_groups[0]['lr']:.6f}"
        if do_eval or (epoch + 1) % 5 == 0 or epoch == 0 or is_last:
            print(log_line, flush=True)

        # Crash-safety: periodic checkpoint so a late crash doesn't lose the model.
        if args.save_model and ((epoch + 1) % args.checkpoint_every == 0 or is_last):
            _save_full_model(args, variant, model, mean, std, feature_names,
                             task_map, input_dim, n_epochs,
                             f"stage2_{variant}_{args.label_scheme}_latest.pt")

    # Save per-epoch test log
    if test_epoch_log and args.save_model:
        save_test_epoch_log(test_epoch_log, f"stage2_{variant}_{args.label_scheme}",
                            args.save_model)

    # Final test evaluation
    if test_loader is not None:
        preds_list, true_list = evaluate_crf(model, test_loader, device)
        metrics = _compute_all_metrics(preds_list, true_list, args.boundary_tolerance)

        preds_flat = np.concatenate(preds_list)
        true_flat = np.concatenate(true_list)
        bundle = metric_bundle(true_flat, preds_flat, n_classes=N_CLASSES, n_boot=2000)

        print(f"\n  Final Test Results:")
        print(f"    Timestep acc:   {metrics['ts_accuracy']:.4f}")
        print(f"    Timestep F1:    {metrics['ts_f1_macro']:.4f}")
        print(f"    Segment F1:     {metrics['segment_f1']:.4f}")
        print(f"    Boundary F1:    {metrics['boundary_f1']:.4f}  "
              f"(P={metrics['boundary_precision']:.4f} R={metrics['boundary_recall']:.4f})")
        print(f"    Agreement (95% CI): "
              f"{format_bundle(bundle, keys=['mcc', 'kappa', 'balanced_accuracy'])}")

        target_names = [task_map[i] for i in range(N_CLASSES)]
        print(classification_report(true_flat, preds_flat,
                                    target_names=target_names, digits=4))

        cm = confusion_matrix(true_flat, preds_flat, labels=list(range(N_CLASSES)))
        print("  Confusion Matrix:")
        print(f"  {'':>10s}  " + "  ".join(f"{s:>7s}" for s in TASK_SHORT))
        for i, row in enumerate(cm):
            print(f"  {TASK_SHORT[i]:>10s}  " + "  ".join(f"{v:7d}" for v in row))

        if args.save_model:
            save_test_results(args.save_model, f"stage2_{variant}_{args.label_scheme}", {
                "model": "unigru_crf", "variant": variant,
                "label_scheme": args.label_scheme, "normalize": args.normalize,
                "ts_accuracy": float(metrics["ts_accuracy"]),
                "ts_f1_macro": float(metrics["ts_f1_macro"]),
                "ts_f1_weighted": float(metrics["ts_f1_weighted"]),
                "segment_f1": float(metrics["segment_f1"]),
                "boundary_f1": float(metrics["boundary_f1"]),
                "boundary_precision": float(metrics["boundary_precision"]),
                "boundary_recall": float(metrics["boundary_recall"]),
                "mcc": float(metrics["mcc"]), "kappa": float(metrics["kappa"]),
                "confusion_matrix": cm.tolist(), "confusion_level": "timestep",
                "labels": [task_map[i] for i in range(N_CLASSES)],
            })

    # Save final model
    if args.save_model:
        date_str = datetime.now().strftime("%Y%m%d")
        var_tag = "A_raw_concatenation" if variant == "A" else "B_feature_stitching"
        filename = f"stage2_{var_tag}_{args.label_scheme}_unigru_crf_{date_str}.pt"
        save_path = _save_full_model(
            args, var_tag, model, mean, std, feature_names, task_map,
            input_dim, n_epochs, filename,
        )
        print(f"  Saved model to {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2: UniGRU + CRF Training")
    parser.add_argument("--variant", choices=["A", "B"], required=True,
                        help="A=raw concatenation, B=feature stitching")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with processed data files")
    parser.add_argument("--test-data", type=str, default=None,
                        help="Path to test pkl (required for full mode)")
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
    parser.add_argument("--epochs", type=int, default=80,
                        help="Max epochs per fold (CV mode)")
    parser.add_argument("--patience", type=int, default=12,
                        help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--boundary-tolerance", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--decay", type=float, default=1.0,
                        help="Exponential decay for cumulative temporal features "
                             "(Variant A only). 1.0 disables it (default); decay did "
                             "not improve cross-source segment-F1 in our tests.")
    parser.add_argument("--n-sessions", type=int, default=2000,
                        help="Sessions to compose per fold (Variant B only)")
    parser.add_argument("--min-tasks", type=int, default=2)
    parser.add_argument("--max-tasks", type=int, default=5)

    parser.add_argument("--label-scheme", choices=["fine", "coarse"], default="fine",
                        help="fine=7 tasks, coarse=3 interaction groups")
    parser.add_argument("--normalize", choices=["global", "per-source"], default="global",
                        help="per-source: Variant A by source, Variant B per-session")
    parser.add_argument("--cv-json", type=str, default=None,
                        help="Write machine-readable LOSO CV summary to this path")
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Full mode: decode the test set every N epochs (cost saver)")
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Full mode: save a latest-checkpoint every N epochs")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Tee stdout to this file (for overnight runs)")
    args = parser.parse_args()

    if args.mode in ("full", "both") and args.test_data is None:
        parser.error("--test-data is required for full/both mode")
    if args.mode == "full" and args.cv_checkpoint is None and args.fixed_epochs is None:
        parser.error("--cv-checkpoint or --fixed-epochs required for full mode")

    if args.log_file:
        sys.stdout = _Tee(sys.stdout, args.log_file)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Activate label scheme: rebind module globals used by helpers/model.
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
    print(f"Device: {device}  |  Variant: {args.variant}  |  "
          f"Label scheme: {args.label_scheme} ({N_CLASSES} classes)  |  "
          f"Normalize: {args.normalize}")

    data_dir = Path(args.data_dir)

    fold_results = None
    input_dim, feature_names, task_map = None, None, None

    if args.mode in ("cv", "both"):
        fold_results, input_dim, feature_names, task_map = run_loso_cv(
            args.variant, data_dir, args, scheme, device, pin_memory, num_workers,
        )
        if args.cv_json:
            write_cv_json(
                args.cv_json, fold_results, model=f"unigru_crf_{args.variant}",
                label_scheme=args.label_scheme, normalize=args.normalize, seed=args.seed,
                metric_keys=["ts_accuracy", "ts_f1_macro", "segment_f1",
                             "boundary_f1", "mcc", "kappa"],
            )

    if args.mode in ("full", "both"):
        n_epochs = resolve_n_epochs(args, fold_results)
        print(f"\n  Using n_epochs={n_epochs} for full training")

        test_data = None
        if args.test_data:
            with open(args.test_data, "rb") as f:
                test_data = pickle.load(f)

        run_full_training(
            args.variant, data_dir, n_epochs, test_data, input_dim,
            feature_names, task_map, args, scheme, device, pin_memory, num_workers,
        )


if __name__ == "__main__":
    main()
