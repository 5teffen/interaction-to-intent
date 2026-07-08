"""Shared spatial computations used across all featurization modules."""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


def compute_cluster_centroids(proj_df: pd.DataFrame) -> dict:
    """Compute centroid and radius for each cluster (c5 column)."""
    centroids = {}
    for c in proj_df["c5"].unique():
        mask = proj_df["c5"] == c
        cx = proj_df.loc[mask, "x"].mean()
        cy = proj_df.loc[mask, "y"].mean()
        dists = np.sqrt(
            (proj_df.loc[mask, "x"] - cx) ** 2
            + (proj_df.loc[mask, "y"] - cy) ** 2
        )
        radius = dists.max() if dists.max() > 0 else 1e-6
        centroids[int(c)] = {"cx": cx, "cy": cy, "radius": radius}
    return centroids


def compute_local_density(
    proj_df: pd.DataFrame, epsilon: float = 0.05
) -> np.ndarray:
    """Count neighbors within epsilon for each point (excluding self)."""
    coords = proj_df[["x", "y"]].values
    tree = KDTree(coords)
    counts = tree.query_ball_point(coords, r=epsilon, return_length=True)
    return counts.astype(float) - 1
