"""Shared constants for the Interactions2Intent pipeline."""

import numpy as np

VALID_DATASETS = {"basketball", "pokemon", "weather", "recipes"}
VALID_TASK_TYPES = {0, 1, 2, 3, 4, 5, 6}
N_CLASSES = 7
N_CLUSTERS = 5

TRAIN_DATASETS = ["basketball", "pokemon", "weather"]
TEST_DATASETS = ["recipes"]

TASK_LABEL_MAP = {
    0: "Generate Clusters",
    1: "Name Cluster",
    2: "Find Similar Points",
    3: "Identify Outliers",
    4: "Add Data to Projection",
    5: "Compare Clusters",
    6: "Map Synthesized to Original Dimension",
}

TASK_SHORT = [
    "GenClust", "NameCl", "FindSim", "IdOutl",
    "AddData", "CompCl", "MapDim",
]

TEMPORAL_FEATURE_NAMES = [
    "norm_x", "norm_y", "local_density", "periphery_score",
    "cluster_0", "cluster_1", "cluster_2", "cluster_3", "cluster_4",
    "step_distance", "bearing_sin", "bearing_cos",
    "turning_sin", "turning_cos", "log_dwell_time",
    "cluster_transition", "cluster_revisit", "revisit_flag",
    "time_since_point_last_visited", "distance_from_first_hover",
    "running_unique_cluster_count", "recent_cluster_entropy",
    "recent_step_distance_mean", "cumulative_coverage",
]

SUMMARY_FEATURE_NAMES = [
    "step_dist_mean", "step_dist_std", "step_dist_median",
    "step_dist_p10", "step_dist_p90",
    "log_dwell_mean", "log_dwell_std", "log_dwell_median",
    "log_dwell_p10", "log_dwell_p90",
    "turn_mag_mean", "turn_mag_std", "turn_mag_median",
    "periph_mean", "periph_std", "periph_median",
    "density_mean", "density_std", "density_median",
    "time_revisit_mean", "time_revisit_std", "time_revisit_max",
    "cluster_frac_0", "cluster_frac_1", "cluster_frac_2",
    "cluster_frac_3", "cluster_frac_4",
    "cluster_transition_rate", "cluster_revisit_rate", "point_revisit_rate",
    "final_coverage", "final_unique_clusters",
    "seq_length", "log_total_duration", "convex_hull_area",
    "dist_first_to_last", "n_cluster_runs",
    "dist_from_first_mean", "dist_from_first_std",
]

STATELESS_FEATURE_NAMES = [
    "norm_x", "norm_y", "local_density", "periphery_score",
    "cluster_0", "cluster_1", "cluster_2", "cluster_3", "cluster_4",
    "step_distance", "bearing_sin", "bearing_cos",
    "turning_sin", "turning_cos", "log_dwell_time",
]

# ── Coarse (3-class) label scheme ─────────────────────────────────────────────
# Theory-defined grouping of the 7 fine tasks into broader interaction patterns.
N_COARSE_CLASSES = 3

COARSE_LABEL_MAP = {
    0: "Local Exploration",     # focused within a single cluster / neighborhood
    1: "Global Scanning",       # widespread movement across the whole plot
    2: "Comparative Probing",   # repetitive movement between clusters / along axis
}

COARSE_SHORT = ["Local", "Global", "Compare"]

# fine task id -> coarse class id
FINE_TO_COARSE = {
    1: 0, 2: 0, 3: 0,   # Name Cluster, Find Similar Points, Identify Outliers
    0: 1, 4: 1,         # Generate Clusters, Add Data to Projection
    5: 2, 6: 2,         # Compare Clusters, Map Synth. to Orig. Dimension
}


# ── Feature groups (for leave-one-group-out ablations) ───────────────────────
# Summary groups partition the 39-dim summary vector by semantic category.
# Summary feature groups, matching the categories in the feature-description
# table (Aggregated statistics / Distributional / Rates / Cumulative / Whole-seq).
SUMMARY_FEATURE_GROUPS = {
    "aggregated_statistics": [
        "step_dist_mean", "step_dist_std", "step_dist_median", "step_dist_p10", "step_dist_p90",
        "log_dwell_mean", "log_dwell_std", "log_dwell_median", "log_dwell_p10", "log_dwell_p90",
        "turn_mag_mean", "turn_mag_std", "turn_mag_median",
        "periph_mean", "periph_std", "periph_median",
        "density_mean", "density_std", "density_median",
        "time_revisit_mean", "time_revisit_std", "time_revisit_max",
    ],
    "distributional": [
        "cluster_frac_0", "cluster_frac_1", "cluster_frac_2", "cluster_frac_3", "cluster_frac_4",
    ],
    "rates": [
        "cluster_transition_rate", "cluster_revisit_rate", "point_revisit_rate",
    ],
    "cumulative": [
        "final_coverage", "final_unique_clusters",
    ],
    "whole_sequence": [
        "seq_length", "log_total_duration", "convex_hull_area",
        "dist_first_to_last", "n_cluster_runs", "dist_from_first_mean", "dist_from_first_std",
    ],
}

# Temporal feature groups, matching the feature-description table.
# Sub-categories: Point-level / Step-level / Anchor-relative / Running statistics.
# Plus the first-tier stateless vs stateful split (the S column): stateless
# includes cluster_transition (it only needs the previous step).
TEMPORAL_FEATURE_GROUPS = {
    # first tier: state dependence
    "stateless": [
        "norm_x", "norm_y", "local_density", "periphery_score",
        "cluster_0", "cluster_1", "cluster_2", "cluster_3", "cluster_4",
        "step_distance", "bearing_sin", "bearing_cos", "turning_sin", "turning_cos",
        "log_dwell_time", "cluster_transition",
    ],
    "stateful": [
        "cluster_revisit", "revisit_flag", "time_since_point_last_visited",
        "distance_from_first_hover", "running_unique_cluster_count",
        "recent_cluster_entropy", "recent_step_distance_mean", "cumulative_coverage",
    ],
    # second tier: feature-table sub-categories
    "point_level": [
        "norm_x", "norm_y", "local_density", "periphery_score",
        "cluster_0", "cluster_1", "cluster_2", "cluster_3", "cluster_4",
    ],
    "step_level": [
        "step_distance", "bearing_sin", "bearing_cos", "turning_sin", "turning_cos",
        "log_dwell_time", "cluster_transition", "cluster_revisit", "revisit_flag",
        "time_since_point_last_visited",
    ],
    "anchor_relative": ["distance_from_first_hover"],
    "running_statistics": [
        "running_unique_cluster_count", "recent_cluster_entropy",
        "recent_step_distance_mean", "cumulative_coverage",
    ],
}


def resolve_group_features(groups: dict, group_name: str) -> list:
    """Return the feature-name list for a group, erroring on unknown names."""
    if group_name not in groups:
        raise ValueError(
            f"Unknown feature group {group_name!r}; "
            f"available: {sorted(groups)}"
        )
    return list(groups[group_name])


def select_feature_indices(feature_names: list, keep=None, drop=None) -> list:
    """
    Return indices into feature_names to retain.

    keep: keep only these feature names (drop everything else).
    drop: drop these feature names (keep everything else).
    Exactly one of keep/drop should be set; if neither, all indices are kept.
    Unknown names raise, to fail loudly on typos.
    """
    names = list(feature_names)
    name_set = set(names)
    if keep is not None and drop is not None:
        raise ValueError("Pass only one of keep/drop")
    if keep is not None:
        missing = [n for n in keep if n not in name_set]
        if missing:
            raise ValueError(f"keep names not in features: {missing}")
        keep_set = set(keep)
        return [i for i, n in enumerate(names) if n in keep_set]
    if drop is not None:
        missing = [n for n in drop if n not in name_set]
        if missing:
            raise ValueError(f"drop names not in features: {missing}")
        drop_set = set(drop)
        return [i for i, n in enumerate(names) if n not in drop_set]
    return list(range(len(names)))


# Monotonic, session-accumulating temporal features. In a single-task atomic
# sequence these are task-local, but in a stitched multi-task session they
# accumulate across task boundaries and end up encoding session position rather
# than the current task. An exponential decay restores task-locality.
CUMULATIVE_FEATURE_NAMES = ["running_unique_cluster_count", "cumulative_coverage"]


def get_label_scheme(name: str) -> dict:
    """
    Return active class metadata and a label-remap function for a scheme.

    name="fine"   -> original 7-class tasks (identity remap)
    name="coarse" -> 3-class interaction groups (remaps fine labels)

    Keys: n_classes, label_map, short, remap(labels)->np.ndarray
    """
    if name == "fine":
        return {
            "n_classes": N_CLASSES,
            "label_map": dict(TASK_LABEL_MAP),
            "short": list(TASK_SHORT),
            "remap": lambda y: np.asarray(y),
        }
    if name == "coarse":
        lut = np.array([FINE_TO_COARSE[i] for i in range(7)])
        return {
            "n_classes": N_COARSE_CLASSES,
            "label_map": dict(COARSE_LABEL_MAP),
            "short": list(COARSE_SHORT),
            "remap": lambda y: lut[np.asarray(y)],
        }
    raise ValueError(f"Unknown label scheme: {name!r} (expected 'fine' or 'coarse')")
