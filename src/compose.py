"""Session composition logic for Stage 2 multi-task sequences."""

import math
import numpy as np
import pandas as pd

from .constants import STATELESS_FEATURE_NAMES

_SL = {name: i for i, name in enumerate(STATELESS_FEATURE_NAMES)}
_IDX_STEP_DISTANCE = _SL["step_distance"]
_IDX_BEARING_SIN = _SL["bearing_sin"]
_IDX_BEARING_COS = _SL["bearing_cos"]
_IDX_TURNING_SIN = _SL["turning_sin"]
_IDX_TURNING_COS = _SL["turning_cos"]
_IDX_LOG_DWELL = _SL["log_dwell_time"]
_IDX_NORM_X = _SL["norm_x"]
_IDX_NORM_Y = _SL["norm_y"]


def compose_session_events(
    atomic_sequences: list[list[dict]],
    transition_gap_ms: float = 3000.0,
) -> list[dict]:
    """
    Concatenate raw event lists, rebasing timestamps so each subsequent
    sequence starts transition_gap_ms after the previous one ends.
    """
    composed = []
    time_offset = 0.0

    for seq_idx, events in enumerate(atomic_sequences):
        if seq_idx > 0:
            time_offset = (
                composed[-1]["timestamp_ms"]
                + transition_gap_ms
                - events[0]["timestamp_ms"]
            )

        for evt in events:
            new_evt = dict(evt)
            new_evt["timestamp_ms"] = evt["timestamp_ms"] + time_offset
            composed.append(new_evt)

    return composed


def compose_session_labels(
    atomic_sequences: list[list[dict]],
    task_types: list[int],
) -> np.ndarray:
    """Create per-timestep label array from atomic sequence lengths and task types."""
    labels = []
    for events, task in zip(atomic_sequences, task_types):
        labels.extend([task] * len(events))
    return np.array(labels, dtype=np.int64)


def stitch_feature_matrices(matrices: list[np.ndarray]) -> np.ndarray:
    """
    Vertically stack pre-computed stateless feature matrices (15 dims each).
    Fix boundary timesteps:
      - step features recomputed from prior segment's last position
      - log_dwell_time zeroed at boundaries
      - turning angle zeroed (no meaningful prior bearing)

    NOTE: Makes a copy of boundary rows to avoid mutating the input arrays
    (fixes bug B4 from the review).
    """
    if len(matrices) == 1:
        return matrices[0].astype(np.float32, copy=False)

    all_rows = []

    for seg_idx, mat in enumerate(matrices):
        if seg_idx > 0:
            mat = mat.copy()

            prev_mat = matrices[seg_idx - 1]
            prev_x = prev_mat[-1, _IDX_NORM_X]
            prev_y = prev_mat[-1, _IDX_NORM_Y]
            curr_x = mat[0, _IDX_NORM_X]
            curr_y = mat[0, _IDX_NORM_Y]

            dx = curr_x - prev_x
            dy = curr_y - prev_y
            step_dist = math.sqrt(dx ** 2 + dy ** 2)
            bearing = math.atan2(dy, dx)

            mat[0, _IDX_STEP_DISTANCE] = step_dist
            mat[0, _IDX_BEARING_SIN] = math.sin(bearing)
            mat[0, _IDX_BEARING_COS] = math.cos(bearing)
            mat[0, _IDX_TURNING_SIN] = 0.0
            mat[0, _IDX_TURNING_COS] = 0.0
            mat[0, _IDX_LOG_DWELL] = 0.0

        all_rows.append(mat)

    return np.vstack(all_rows).astype(np.float32)


def generate_sessions_same_projection(
    df: pd.DataFrame,
    n_sessions: int,
    tasks_per_session: tuple[int, int],
    rng: np.random.Generator,
) -> list[dict]:
    """
    Generate composed sessions by sampling atomic sequences from the same
    (dataset, projection) pair. Used by Variant A.

    Returns list of dicts with keys: dataset, projection, atomic_indices, task_types.
    """
    groups: dict[tuple[str, str], list[int]] = {}
    for idx, row in df.iterrows():
        key = (row["dataset"], row["projection"])
        groups.setdefault(key, []).append(idx)

    keys = list(groups.keys())
    sessions = []

    for _ in range(n_sessions):
        key = keys[rng.integers(len(keys))]
        available = groups[key]
        n_tasks = rng.integers(tasks_per_session[0], tasks_per_session[1] + 1)
        n_tasks = min(n_tasks, len(available))
        chosen_indices = rng.choice(available, size=n_tasks, replace=True).tolist()
        task_types = [int(df.loc[i, "task_type"]) for i in chosen_indices]

        sessions.append({
            "dataset": key[0],
            "projection": key[1],
            "atomic_indices": chosen_indices,
            "task_types": task_types,
        })

    return sessions


def generate_sessions_any(
    n_sequences: int,
    n_sessions: int,
    tasks_per_session: tuple[int, int],
    rng: np.random.Generator,
) -> list[list[int]]:
    """
    Generate session compositions by sampling atomic sequence indices
    without (dataset, projection) constraint. Used by Variant B.
    """
    sessions = []
    for _ in range(n_sessions):
        n_tasks = rng.integers(tasks_per_session[0], tasks_per_session[1] + 1)
        chosen = rng.choice(n_sequences, size=n_tasks, replace=True).tolist()
        sessions.append(chosen)
    return sessions
