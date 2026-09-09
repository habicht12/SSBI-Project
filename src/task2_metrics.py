"""Neighbour overlap for aligned observations in Task 2 embeddings."""

import numpy as np
from sklearn import config_context
from sklearn.neighbors import NearestNeighbors


def neighbor_indices(values, k=15):
    """Return exactly k neighbours per row, excluding the row itself."""
    values = np.asarray(values)
    if values.ndim != 2 or not 1 <= k < len(values):
        raise ValueError("Expected a matrix and 1 <= k < number of observations.")
    with config_context(working_memory=64):
        # With X=None sklearn excludes each indexed observation automatically.
        return NearestNeighbors(n_neighbors=k, n_jobs=1).fit(values).kneighbors(
            return_distance=False
        )


def neighborhood_overlap(indices_a, indices_b):
    """Mean fraction of shared neighbours for two aligned index matrices."""
    if indices_a.shape != indices_b.shape or indices_a.ndim != 2:
        raise ValueError("Neighbour matrices must have the same two-dimensional shape.")
    k = indices_a.shape[1]
    if k == 0:
        raise ValueError("At least one neighbour is required.")
    return float(np.mean([
        len(set(a).intersection(b)) / k for a, b in zip(indices_a, indices_b)
    ]))


def knn_preservation(values_a, values_b, k=15):
    """Symmetric kNN recall for identical, identically ordered observations."""
    if len(values_a) != len(values_b):
        raise ValueError("Representations must contain the same aligned observations.")
    return neighborhood_overlap(neighbor_indices(values_a, k), neighbor_indices(values_b, k))
