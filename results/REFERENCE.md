# Reference results

Expected numbers for confirming a successful reproduction. All values use the reported
setting: **per-source normalization**, seed 42, leave-one-source-out (LOSO) cross-validation
over the three training sources (basketball, pokemon, weather) with **recipes** held out as
the test source. Small deviations (±0.01–0.02) are expected across hardware/library versions;
the ablation drops are single-seed and noisy below ~0.02.

Chance baselines: 7-class acc ≈ 0.143; 3-class majority (Local) acc ≈ 0.441.

## Table 1 — Atomic task classification

LOSO cross-validation (mean ± fold std):

| Model | Scheme | Acc | F1-macro | κ |
|---|---|---|---|---|
| XGBoost (summary) | 7-class | 0.549 ± 0.029 | 0.548 ± 0.028 | 0.474 |
| BiGRU (temporal)  | 7-class | 0.568 ± 0.008 | 0.567 ± 0.010 | 0.495 |
| XGBoost (summary) | 3-class | 0.678 ± 0.013 | 0.659 ± 0.016 | 0.500 |
| BiGRU (temporal)  | 3-class | 0.692 ± 0.015 | 0.685 ± 0.014 | 0.532 |

Held-out test (recipes):

| Model | Scheme | Acc | F1-macro | κ |
|---|---|---|---|---|
| XGBoost | 7-class | 0.621 | 0.624 | 0.556 |
| BiGRU   | 7-class | 0.598 | 0.602 | 0.531 |
| XGBoost | 3-class | 0.690 | 0.678 | 0.519 |
| BiGRU   | 3-class | 0.730 | 0.720 | 0.588 |

## Table 1 — Online multi-task classification (UniGRU-CRF)

Metrics: Seg = segment-F1 (primary), TS = timestep-F1, Bnd = boundary-F1 (±3 steps).

| Stitching | Scheme | Split | Seg | TS | Bnd |
|---|---|---|---|---|---|
| Log-level (Variant A)     | 7-class | CV   | 0.399 ± 0.023 | 0.378 | 0.172 |
| Feature-level (Variant B) | 7-class | CV   | 0.353 ± 0.019 | 0.338 | 0.394 |
| Log-level (Variant A)     | 7-class | Test | 0.445 | 0.426 | 0.150 |
| Feature-level (Variant B) | 7-class | Test | 0.325 | 0.288 | 0.401 |
| Log-level (Variant A)     | 3-class | CV   | 0.542 ± 0.024 | 0.528 | 0.125 |
| Feature-level (Variant B) | 3-class | CV   | 0.562 ± 0.011 | 0.529 | 0.323 |
| Log-level (Variant A)     | 3-class | Test | 0.511 | 0.500 | 0.105 |
| Feature-level (Variant B) | 3-class | Test | 0.497 | 0.433 | 0.302 |

## Table 2 — Feature ablations (LOSO, ΔF1-macro vs. full model)

Full-model F1-macro — XGBoost: 0.545 (7-class) / 0.655 (3-class);
BiGRU: 0.567 (7-class) / 0.681 (3-class).

XGBoost, drop one feature group:

| Dropped group | Δ 7-class | Δ 3-class |
|---|---|---|
| kinematics          | −0.037 | −0.011 |
| cluster_composition | −0.030 | −0.014 |
| trajectory_geometry | −0.026 | −0.028 |
| dwell               | −0.024 | +0.004 |
| cluster_dynamics    | −0.021 | +0.003 |
| coverage            | −0.013 | +0.000 |
| revisit_timing      | −0.004 | −0.002 |
| spatial_context     | +0.001 | +0.004 |

BiGRU temporal ablation:

| Configuration | Δ 7-class | Δ 3-class |
|---|---|---|
| − stateful features         | −0.096 | −0.081 |
| shuffled timestep order (H4)| −0.043 | −0.017 |
| − stateless features        | −0.024 | −0.015 |

Reproduce with: `make table1` (both tables' classification rows) and `make table2` (ablations).
