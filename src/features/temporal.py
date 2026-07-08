"""
Temporal featurization: per-timestep (T, 24) feature matrices.

This is the single canonical implementation used by both Stage 1 (atomic
sequences) and Stage 2 Variant A (composed sessions after event concatenation).
"""

import math
import numpy as np
import pandas as pd
from collections import Counter


def featurize_sequence(
    events: list[dict],
    proj_df: pd.DataFrame,
    centroids: dict,
    densities: np.ndarray,
    recent_window: int = 15,
    coverage_grid_size: int = 10,
) -> np.ndarray:
    """
    Convert a sequence of hover events into a (T, 24) feature matrix.

    Features:
      Point-level (9):  norm_x, norm_y, local_density, periphery_score, cluster_onehot(5)
      Step-level (10):  step_distance, bearing_sin/cos, turning_sin/cos,
                        log_dwell_time, cluster_transition, cluster_revisit,
                        revisit_flag, time_since_point_last_visited
      Anchor (1):       distance_from_first_hover
      Running (4):      unique_cluster_count, recent_cluster_entropy,
                        recent_step_distance_mean, cumulative_coverage
    """
    T = len(events)
    features = np.zeros((T, 24), dtype=np.float32)

    x_min, x_max = proj_df["x"].min(), proj_df["x"].max()
    y_min, y_max = proj_df["y"].min(), proj_df["y"].max()
    x_range = x_max - x_min if x_max - x_min > 0 else 1e-6
    y_range = y_max - y_min if y_max - y_min > 0 else 1e-6

    coverage_grid: set[tuple[int, int]] = set()
    visited_clusters: set[int] = set()
    visited_points: dict[int, float] = {}
    prev_bearing: float | None = None
    positions_so_far: list[tuple[float, float]] = []
    step_distances_recent: list[float] = []

    for t, evt in enumerate(events):
        pid = evt["point_id"]
        x_raw = evt["x"]
        y_raw = evt["y"]
        cluster = evt["cluster"]
        ts = evt["timestamp_ms"]

        x_norm = (x_raw - x_min) / x_range
        y_norm = (y_raw - y_min) / y_range
        features[t, 0] = x_norm
        features[t, 1] = y_norm

        features[t, 2] = densities[pid]

        if cluster in centroids:
            c_info = centroids[cluster]
            dist_to_centroid = math.sqrt(
                (x_raw - c_info["cx"]) ** 2 + (y_raw - c_info["cy"]) ** 2
            )
            features[t, 3] = dist_to_centroid / c_info["radius"]
        else:
            features[t, 3] = 0.0

        if 0 <= cluster <= 4:
            features[t, 4 + cluster] = 1.0

        if t > 0:
            prev_evt = events[t - 1]
            prev_x = (prev_evt["x"] - x_min) / x_range
            prev_y = (prev_evt["y"] - y_min) / y_range

            dx = x_norm - prev_x
            dy = y_norm - prev_y
            step_dist = math.sqrt(dx ** 2 + dy ** 2)
            features[t, 9] = step_dist

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

            prev_cluster = prev_evt["cluster"]
            cluster_changed = int(cluster != prev_cluster)
            features[t, 15] = cluster_changed
            features[t, 16] = 1.0 if (cluster_changed and cluster in visited_clusters) else 0.0
            features[t, 17] = 1.0 if pid in visited_points else 0.0

            if pid in visited_points:
                time_since = ts - visited_points[pid]
                features[t, 18] = math.log1p(max(time_since, 0))
            else:
                features[t, 18] = 0.0
        else:
            prev_bearing = None
            features[t, 9:19] = 0.0

        visited_clusters.add(cluster)
        visited_points[pid] = ts
        positions_so_far.append((x_norm, y_norm))

        x0, y0 = positions_so_far[0]
        features[t, 19] = math.sqrt((x_norm - x0) ** 2 + (y_norm - y0) ** 2)

        features[t, 20] = len(visited_clusters)

        start = max(0, t - recent_window + 1)
        recent_clusters = [events[i]["cluster"] for i in range(start, t + 1)]
        counts = Counter(recent_clusters)
        total = sum(counts.values())
        entropy = 0.0
        for cnt in counts.values():
            p = cnt / total
            if p > 0:
                entropy -= p * math.log(p)
        features[t, 21] = entropy

        if t > 0:
            step_distances_recent.append(features[t, 9])
            if len(step_distances_recent) > recent_window:
                step_distances_recent.pop(0)
            features[t, 22] = float(np.mean(step_distances_recent))
        else:
            features[t, 22] = 0.0

        gx = int(x_norm * (coverage_grid_size - 1))
        gy = int(y_norm * (coverage_grid_size - 1))
        gx = max(0, min(gx, coverage_grid_size - 1))
        gy = max(0, min(gy, coverage_grid_size - 1))
        coverage_grid.add((gx, gy))
        features[t, 23] = len(coverage_grid) / (coverage_grid_size ** 2)

    return features
