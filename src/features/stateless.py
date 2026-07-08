"""Stateless per-timestep featurization (15 dims) for Stage 2 Variant B."""

import math
import numpy as np
import pandas as pd


def featurize_stateless(
    events: list[dict],
    proj_df: pd.DataFrame,
    centroids: dict,
    densities: np.ndarray,
) -> np.ndarray:
    """
    Compute only stateless per-timestep features (15 dims).
    No cumulative state, no revisit tracking, no windowed stats.

    Features (15):
        0-1: norm_x, norm_y
        2:   local_density
        3:   periphery_score
        4-8: cluster_onehot (5)
        9:   step_distance
        10-11: bearing_sin, bearing_cos
        12-13: turning_sin, turning_cos
        14:  log_dwell_time
    """
    T = len(events)
    features = np.zeros((T, 15), dtype=np.float32)

    x_min, x_max = proj_df["x"].min(), proj_df["x"].max()
    y_min, y_max = proj_df["y"].min(), proj_df["y"].max()
    x_range = x_max - x_min if x_max - x_min > 0 else 1e-6
    y_range = y_max - y_min if y_max - y_min > 0 else 1e-6

    prev_bearing = None

    for t, evt in enumerate(events):
        pid = evt["point_id"]
        x_raw, y_raw = evt["x"], evt["y"]
        cluster = evt["cluster"]
        ts = evt["timestamp_ms"]

        x_norm = (x_raw - x_min) / x_range
        y_norm = (y_raw - y_min) / y_range
        features[t, 0] = x_norm
        features[t, 1] = y_norm
        features[t, 2] = densities[pid]

        if cluster in centroids:
            c_info = centroids[cluster]
            d = math.sqrt((x_raw - c_info["cx"]) ** 2 + (y_raw - c_info["cy"]) ** 2)
            features[t, 3] = d / c_info["radius"]

        if 0 <= cluster <= 4:
            features[t, 4 + cluster] = 1.0

        if t > 0:
            prev_evt = events[t - 1]
            prev_x = (prev_evt["x"] - x_min) / x_range
            prev_y = (prev_evt["y"] - y_min) / y_range

            dx = x_norm - prev_x
            dy = y_norm - prev_y
            features[t, 9] = math.sqrt(dx ** 2 + dy ** 2)

            bearing = math.atan2(dy, dx)
            features[t, 10] = math.sin(bearing)
            features[t, 11] = math.cos(bearing)

            if prev_bearing is not None:
                delta = bearing - prev_bearing
                delta = (delta + math.pi) % (2 * math.pi) - math.pi
                features[t, 12] = math.sin(delta)
                features[t, 13] = math.cos(delta)
            else:
                features[t, 12] = 0.0
                features[t, 13] = 0.0
            prev_bearing = bearing

            gap_ms = ts - prev_evt["timestamp_ms"]
            features[t, 14] = math.log1p(max(gap_ms, 0))
        else:
            prev_bearing = None
            features[t, 9:15] = 0.0

    return features
