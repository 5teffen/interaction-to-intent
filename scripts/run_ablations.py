#!/usr/bin/env python3
"""
Ablation / robustness orchestrator for Stage 1.

Drives the existing training scripts over a matrix of {config x seed}, reads the
machine-readable LOSO CV summary each run emits (--cv-json), and tabulates
macro-F1 with mean +/- std across seeds plus the drop-off versus the full model.

It never duplicates the training logic: each run is a normal subprocess call to
train_stage1_{xgboost,bigru}.py in --mode cv, so results are identical to a
hand-run command.

Modes:
    multiseed  full model only, across seeds (robustness numbers)
    ablation   full + leave-one-group-out for every feature group (drop-off)
    shuffle    full vs timestep-shuffled (BiGRU only, H4 test)
    all        union of the above applicable to the model

Examples:
    # Robustness across 5 seeds, coarse + per-source BiGRU
    python scripts/run_ablations.py --model bigru --mode multiseed \
        --label-scheme coarse --normalize per-source --seeds 42,43,44,45,46

    # Feature-group ablation for XGBoost (fine labels)
    python scripts/run_ablations.py --model xgboost --mode ablation \
        --seeds 42,43,44 --out results/ablation_xgb_fine.md

    # H4 temporal-order test for BiGRU
    python scripts/run_ablations.py --model bigru --mode shuffle \
        --label-scheme coarse --normalize per-source --seeds 42,43,44
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from src.constants import SUMMARY_FEATURE_GROUPS, TEMPORAL_FEATURE_GROUPS

MODEL_SCRIPT = {
    "xgboost": "scripts/train_stage1_xgboost.py",
    "bigru": "scripts/train_stage1_bigru.py",
}
MODEL_DATA = {
    "xgboost": "train_summary.pkl",
    "bigru": "train_temporal.pkl",
}
MODEL_GROUPS = {
    "xgboost": SUMMARY_FEATURE_GROUPS,
    "bigru": TEMPORAL_FEATURE_GROUPS,
}


# ── Config matrix ────────────────────────────────────────────────────────────

def build_configs(model: str, mode: str) -> list[tuple[str, list[str]]]:
    """Return [(config_name, extra_cli_args), ...]. 'full' is always first."""
    configs = [("full", [])]
    groups = sorted(MODEL_GROUPS[model])

    want_ablation = mode in ("ablation", "all")
    want_shuffle = mode in ("shuffle", "all") and model == "bigru"

    if want_ablation:
        for g in groups:
            configs.append((f"drop:{g}", ["--drop-group", g]))
    if want_shuffle:
        configs.append(("shuffle_time", ["--shuffle-time"]))

    return configs


# ── Running ──────────────────────────────────────────────────────────────────

def run_config(python: str, script: str, data: str, label_scheme: str,
               normalize: str, extra_args: list[str], seed: int,
               json_path: Path, dry_run: bool = False) -> dict | None:
    """Run one CV config at one seed; return the parsed cv-json (or None)."""
    cmd = [
        python, script,
        "--data", data,
        "--mode", "cv",
        "--label-scheme", label_scheme,
        "--normalize", normalize,
        "--seed", str(seed),
        "--cv-json", str(json_path),
        *extra_args,
    ]
    if dry_run:
        print("  DRY:", " ".join(cmd))
        return None

    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ! run failed (seed={seed}): {' '.join(extra_args) or 'full'}")
        print("  ---- stderr tail ----")
        print("\n".join(proc.stderr.strip().splitlines()[-15:]))
        return None
    with open(json_path) as f:
        return json.load(f)


# ── Aggregation (pure, unit-testable) ────────────────────────────────────────

def summarize_runs(run_jsons: list[dict], metric: str = "f1_macro") -> dict:
    """
    Aggregate per-seed CV summaries for one config.

    Each run contributes its across-fold mean for `metric`. We report the mean
    and std of those per-seed values (seed variance), plus the mean across-fold
    std (fold/source variance), keeping the two sources of variance distinct.
    """
    seed_means = [r["summary"][metric]["mean"] for r in run_jsons]
    fold_stds = [r["summary"][metric]["std"] for r in run_jsons]
    return {
        "n_seeds": len(run_jsons),
        "mean": float(np.mean(seed_means)),
        "seed_std": float(np.std(seed_means)),
        "fold_std": float(np.mean(fold_stds)),
        "seed_means": seed_means,
    }


def build_rows(results: dict[str, list[dict]], metric: str = "f1_macro",
               baseline: str = "full") -> list[dict]:
    """Turn {config: [jsons]} into table rows with drop-off vs baseline."""
    base = summarize_runs(results[baseline], metric) if results.get(baseline) else None
    rows = []
    for name, jsons in results.items():
        if not jsons:
            rows.append({"config": name, "n_seeds": 0})
            continue
        s = summarize_runs(jsons, metric)
        delta = (s["mean"] - base["mean"]) if base else float("nan")
        rows.append({
            "config": name,
            "n_seeds": s["n_seeds"],
            "mean": s["mean"],
            "seed_std": s["seed_std"],
            "fold_std": s["fold_std"],
            "delta": delta,
        })
    # full first, then most-damaging ablations (most negative delta) first
    rows.sort(key=lambda r: (r["config"] != baseline, r.get("delta", 0.0)))
    return rows


def format_markdown(rows: list[dict], metric: str, title: str) -> str:
    lines = [
        f"### {title}",
        "",
        f"| config | n_seeds | {metric} (mean +/- seed_std) | fold_std | delta vs full |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("n_seeds", 0) == 0:
            lines.append(f"| {r['config']} | 0 | (failed) | | |")
            continue
        delta = r["delta"]
        delta_s = "-" if r["config"] == "full" else f"{delta:+.4f}"
        lines.append(
            f"| {r['config']} | {r['n_seeds']} | "
            f"{r['mean']:.4f} +/- {r['seed_std']:.4f} | "
            f"{r['fold_std']:.4f} | {delta_s} |"
        )
    return "\n".join(lines)


# ── Persistence ──────────────────────────────────────────────────────────────

def persist(results, out: Path, metric: str, title: str):
    """Rewrite the markdown table and a raw JSON sidecar from results so far."""
    rows = build_rows(results, metric=metric)
    out.write_text(format_markdown(rows, metric, title) + "\n")
    raw = {"title": title, "metric": metric,
           "results": {k: [r.get("summary", {}) | {"seed": r.get("seed")} for r in v]
                       for k, v in results.items()},
           "rows": rows}
    out.with_suffix(".json").write_text(json.dumps(raw, indent=2))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Stage 1 ablation / robustness runner")
    p.add_argument("--model", choices=["xgboost", "bigru"], required=True)
    p.add_argument("--mode", choices=["multiseed", "ablation", "shuffle", "all"],
                   default="ablation")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--data", default=None, help="Override train pkl path")
    p.add_argument("--label-scheme", choices=["fine", "coarse"], default="fine")
    p.add_argument("--normalize", choices=["global", "per-source"], default="global")
    p.add_argument("--seeds", default="42",
                   help="Seeds for the main (full) config; one is sufficient "
                        "(seed variance is negligible)")
    p.add_argument("--ablation-seeds", type=int, default=1,
                   help="How many of --seeds to use for each ablation config (default 1)")
    p.add_argument("--metric", default="f1_macro",
                   choices=["f1_macro", "accuracy", "f1_weighted"])
    p.add_argument("--out", default=None, help="Markdown table path (JSON sidecar alongside)")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    script = MODEL_SCRIPT[args.model]
    data = args.data or str(Path(args.data_dir) / MODEL_DATA[args.model])
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ablation_seeds = seeds[:max(1, args.ablation_seeds)]
    configs = build_configs(args.model, args.mode)

    out = Path(args.out) if args.out else Path(
        f"results/ablation_{args.model}_{args.label_scheme}_{args.normalize}_{args.mode}.md")
    runs_dir = out.parent / (out.stem + "_runs")

    title = (f"{args.model} | {args.label_scheme} | {args.normalize} | mode={args.mode} | "
             f"full_seeds={seeds} | ablation_seeds={ablation_seeds}")
    print(f"\n=== Ablation run: {title} ===")
    print(f"  configs: {[c[0] for c in configs]}")
    print(f"  output: {out}  (results saved after every run)")

    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {}
    for name, extra in configs:
        # The main (full) config gets all seeds; ablations get a subset.
        cfg_seeds = seeds if name == "full" else ablation_seeds
        runs = []
        for seed in cfg_seeds:
            jp = runs_dir / f"{args.model}_{name.replace(':', '_')}_{seed}.json"
            print(f"  running {name} (seed={seed}) ...", flush=True)
            r = run_config(
                args.python, script, data, args.label_scheme,
                args.normalize, extra, seed, jp, dry_run=args.dry_run,
            )
            if r is not None:
                r["seed"] = seed
                runs.append(r)
                results[name] = runs
                # Persist after every successful run so a crash keeps progress.
                persist(results, out, args.metric, title)

    if args.dry_run:
        return

    rows = build_rows(results, metric=args.metric)
    print("\n" + format_markdown(rows, args.metric, title) + "\n")
    print(f"  Table: {out}  |  raw JSON: {out.with_suffix('.json')}  |  "
          f"per-run: {runs_dir}/")


if __name__ == "__main__":
    main()
