"""Summary featurization: fixed-length (39,) feature vectors per sequence."""

import math
import numpy as np
from collections import Counter
from scipy.spatial import ConvexHull


def convex_hull_area(points: np.ndarray) -> float:
    """Compute the convex hull area of a 2-D point set."""
    if len(points) < 3:
        return 0.0
    unique = np.unique(points, axis=0)
    if len(unique) < 3:
        return 0.0
    try:
        return float(ConvexHull(unique).volume)
    except Exception:
        return 0.0


def _percentile_stats(arr: np.ndarray) -> list[float]:
    """Return [mean, std, median, p10, p90]."""
    if len(arr) == 0:
        return [0.0] * 5
    return [
        float(np.mean(arr)), float(np.std(arr)), float(np.median(arr)),
        float(np.percentile(arr, 10)), float(np.percentile(arr, 90)),
    ]


def _three_stats(arr: np.ndarray) -> list[float]:
    """Return [mean, std, median]."""
    if len(arr) == 0:
        return [0.0] * 3
    return [float(np.mean(arr)), float(np.std(arr)), float(np.median(arr))]


def _three_stats_max(arr: np.ndarray) -> list[float]:
    """Return [mean, std, max]."""
    if len(arr) == 0:
        return [0.0] * 3
    return [float(np.mean(arr)), float(np.std(arr)), float(np.max(arr))]


def featurize_sequence_summary(
    events: list[dict],
    proj_df,
    centroids: dict,
    densities: np.ndarray,
    coverage_grid_size: int = 10,
) -> np.ndarray:
    """Convert a sequence of hover events into a (39,) summary feature vector."""
    T = len(events)

    x_min, x_max = proj_df["x"].min(), proj_df["x"].max()
    y_min, y_max = proj_df["y"].min(), proj_df["y"].max()
    x_range = x_max - x_min if x_max - x_min > 0 else 1e-6
    y_range = y_max - y_min if y_max - y_min > 0 else 1e-6

    step_distances = []
    log_dwell_times = []
    turning_angle_mags = []
    periphery_scores = []
    local_densities = []
    time_since_revisits = []
    cluster_labels = []
    positions_norm = []
    distances_from_first = []

    visited_points: dict[int, float] = {}
    visited_clusters: set[int] = set()
    cluster_transitions = 0
    cluster_revisits = 0
    point_revisits = 0
    prev_bearing = None

    for t, evt in enumerate(events):
        pid = evt["point_id"]
        x_raw, y_raw = evt["x"], evt["y"]
        cluster = evt["cluster"]
        ts = evt["timestamp_ms"]

        x_norm = (x_raw - x_min) / x_range
        y_norm = (y_raw - y_min) / y_range
        positions_norm.append((x_norm, y_norm))

        if cluster in centroids:
            c_info = centroids[cluster]
            d = math.sqrt((x_raw - c_info["cx"]) ** 2 + (y_raw - c_info["cy"]) ** 2)
            periphery_scores.append(d / c_info["radius"])
        else:
            periphery_scores.append(0.0)

        local_densities.append(densities[pid])
        cluster_labels.append(cluster)

        if t > 0:
            prev_evt = events[t - 1]
            prev_x = (prev_evt["x"] - x_min) / x_range
            prev_y = (prev_evt["y"] - y_min) / y_range
            dx, dy = x_norm - prev_x, y_norm - prev_y
            step_distances.append(math.sqrt(dx ** 2 + dy ** 2))

            gap_ms = ts - prev_evt["timestamp_ms"]
            log_dwell_times.append(math.log1p(max(gap_ms, 0)))

            bearing = math.atan2(dy, dx)
            if prev_bearing is not None:
                delta = bearing - prev_bearing
                delta = (delta + math.pi) % (2 * math.pi) - math.pi
                turning_angle_mags.append(abs(delta))
            prev_bearing = bearing

            if cluster != prev_evt["cluster"]:
                cluster_transitions += 1
                if cluster in visited_clusters:
                    cluster_revisits += 1

            if pid in visited_points:
                point_revisits += 1
                time_since_revisits.append(math.log1p(max(ts - visited_points[pid], 0)))

        visited_clusters.add(cluster)
        visited_points[pid] = ts

        x0, y0 = positions_norm[0]
        distances_from_first.append(math.sqrt((x_norm - x0) ** 2 + (y_norm - y0) ** 2))

    feat: list[float] = []
    feat.extend(_percentile_stats(np.array(step_distances)))
    feat.extend(_percentile_stats(np.array(log_dwell_times)))
    feat.extend(_three_stats(np.array(turning_angle_mags) if turning_angle_mags else np.array([0.0])))
    feat.extend(_three_stats(np.array(periphery_scores)))
    feat.extend(_three_stats(np.array(local_densities)))
    feat.extend(_three_stats_max(np.array(time_since_revisits) if time_since_revisits else np.array([0.0])))

    cluster_counts = Counter(cluster_labels)
    for c in range(5):
        feat.append(cluster_counts.get(c, 0) / T)

    n_steps = max(T - 1, 1)
    feat.append(cluster_transitions / n_steps)
    feat.append(cluster_revisits / n_steps)
    feat.append(point_revisits / n_steps)

    coverage_grid: set[tuple[int, int]] = set()
    for xn, yn in positions_norm:
        gx = max(0, min(int(xn * (coverage_grid_size - 1)), coverage_grid_size - 1))
        gy = max(0, min(int(yn * (coverage_grid_size - 1)), coverage_grid_size - 1))
        coverage_grid.add((gx, gy))
    feat.append(len(coverage_grid) / (coverage_grid_size ** 2))
    feat.append(len(visited_clusters))

    feat.append(T)
    total_duration_ms = events[-1]["timestamp_ms"] - events[0]["timestamp_ms"]
    feat.append(math.log1p(max(total_duration_ms, 0)))
    feat.append(convex_hull_area(np.array(positions_norm)))

    x_first, y_first = positions_norm[0]
    x_last, y_last = positions_norm[-1]
    feat.append(math.sqrt((x_last - x_first) ** 2 + (y_last - y_first) ** 2))

    n_runs = 1
    for i in range(1, len(cluster_labels)):
        if cluster_labels[i] != cluster_labels[i - 1]:
            n_runs += 1
    feat.append(n_runs)

    dff = np.array(distances_from_first)
    feat.append(float(dff.mean()))
    feat.append(float(dff.std()))

    return np.array(feat, dtype=np.float32)
