"""Data loading utilities for projected CSVs and spatial statistics."""

import numpy as np
import pandas as pd
from pathlib import Path

from .constants import VALID_DATASETS
from .features.common import compute_cluster_centroids, compute_local_density


def load_projected_data(data_dir: Path) -> dict[tuple[str, str], pd.DataFrame]:
    """Load projected data CSVs into a lookup dict keyed by (dataset, projection)."""
    proj_data = {}
    for dataset in VALID_DATASETS:
        for projection in ("pca", "tsne", "umap"):
            fname = f"{dataset}_projected_data_{projection}.csv"
            fpath = data_dir / fname
            if fpath.exists():
                proj_data[(dataset, projection)] = pd.read_csv(fpath)
    return proj_data


def precompute_spatial_stats(
    proj_data: dict[tuple[str, str], pd.DataFrame],
    epsilon: float = 0.05,
) -> tuple[dict, dict]:
    """
    Precompute cluster centroids and normalized local densities for each
    (dataset, projection) pair.

    Returns:
        (centroids_cache, density_cache) where each is a dict keyed by
        (dataset, projection).
    """
    centroids_cache = {}
    density_cache = {}

    for key, proj_df in proj_data.items():
        centroids_cache[key] = compute_cluster_centroids(proj_df)
        d = compute_local_density(proj_df, epsilon=epsilon)
        d_min, d_max = d.min(), d.max()
        density_cache[key] = (d - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(d)

    return centroids_cache, density_cache
