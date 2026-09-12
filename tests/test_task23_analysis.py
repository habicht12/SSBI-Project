"""Gezielte Tests ohne Benchmarktraining; synthetische Daten und kurze Modell-Smokes."""

import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.task23_analysis import (
    fit_projection, neighbor_indices, neighbor_jaccard, projection_configs,
    recommend_projections, sample_event_indices,
    fit_clusterings, make_neighbor_graph, silhouette_evaluation, select_clusterings,
    cluster_profiles,
)


class SamplingAndMetricTests(unittest.TestCase):
    def test_sampling_is_balanced_reproducible_and_without_replacement(self):
        first = sample_event_indices(1000, 0)
        assert_array_equal(first, sample_event_indices(1000, 0))
        self.assertEqual(len(np.unique(first)), 500)
        self.assertTrue(np.all(np.diff(first) > 0))
        self.assertTrue(np.all((first >= 0) & (first < 1000)))
        self.assertFalse(np.array_equal(first, sample_event_indices(1000, 1)))
        with self.assertRaises(ValueError):
            sample_event_indices(499, 0)

    def test_neighbors_match_hand_computed_distances(self):
        values = np.array([[0.0], [1.0], [4.0], [10.0]])
        assert_array_equal(neighbor_indices(values, 2), [[1, 2], [0, 2], [1, 0], [2, 1]])
        duplicated = neighbor_indices(np.zeros((6, 2)), 2)
        self.assertTrue(all(i not in neighbors for i, neighbors in enumerate(duplicated)))
        self.assertTrue(all(len(set(neighbors)) == 2 for neighbors in duplicated))

    def test_jaccard_sets_and_donor_weighting(self):
        first = np.array([[1, 2], [0, 2], [0, 1]])
        second = np.array([[2, 1], [0, 3], [3, 4]])
        donors = ["a", "a", "b"]
        # Cell scores are 1, 1/3 and 0; donor means are 2/3 and 0.
        self.assertAlmostEqual(neighbor_jaccard(first, second, donors), 1 / 3)
        self.assertEqual(neighbor_jaccard(first, first, donors), 1)
        self.assertEqual(neighbor_jaccard(first, second, donors),
                         neighbor_jaccard(second, first, donors))

    def test_jaccard_invariant_under_rigid_transform_and_uniform_scaling(self):
        values = np.random.default_rng(42).normal(size=(60, 2))
        changed = 3 * values[:, ::-1] + [9, -4]
        with threadpool_limits(limits=1):
            first, second = neighbor_indices(values), neighbor_indices(changed)
        self.assertEqual(neighbor_jaccard(first, second, ["a"] * 60), 1)

    def test_recommendation_ties_prefer_baseline_then_grid_order(self):
        rows = [{"method": c["method"], "variant": c["variant"], "reference_jaccard": 0.2}
                for c in projection_configs()]
        summary = pd.DataFrame(rows)
        selected = recommend_projections(summary)
        self.assertEqual(selected.variant.tolist(), ["pca", "tsne_p30", "umap_n15_d0.1"])
        summary.loc[summary.variant == "pca_white", "reference_jaccard"] += 0.01
        summary.loc[summary.variant == "tsne_p30", "reference_jaccard"] -= 0.01
        self.assertEqual(recommend_projections(summary).variant.tolist(),
                         ["pca_white", "tsne_p5", "umap_n15_d0.1"])


class ProjectionSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = StandardScaler().fit_transform(np.random.default_rng(42).normal(size=(90, 6)))

    def test_pca_whitening_only_rescales_axes(self):
        configs = projection_configs()
        plain, info = fit_projection(self.values, configs[0])
        white, white_info = fit_projection(self.values, configs[1])
        assert_allclose(white, plain / plain.std(axis=0, ddof=1), atol=1e-12)
        self.assertEqual(info, white_info)

    def test_tsne_and_umap_isolated_deterministic_smokes(self):
        for method in ("t-SNE", "UMAP"):
            config = next(c for c in projection_configs() if c["method"] == method)
            with self.subTest(method=method), threadpool_limits(limits=1):
                first, _ = fit_projection(self.values, config)
                second, _ = fit_projection(self.values, config)
                self.assertEqual(first.shape, (90, 2))
                self.assertTrue(np.isfinite(first).all())
                assert_allclose(first, second, rtol=0, atol=0)


class ClusteringTests(unittest.TestCase):
    def test_neighbor_graph_is_simple_undirected_union(self):
        graph = make_neighbor_graph(np.array([[0.0], [1.0], [4.0], [10.0]]), k=1)
        self.assertFalse(graph.is_directed())
        self.assertTrue(graph.is_simple())
        self.assertEqual(set(graph.get_edgelist()), {(0, 1), (1, 2), (2, 3)})

    def test_all_clustering_methods_small_smoke(self):
        values = np.random.default_rng(42).normal(size=(90, 6))
        with threadpool_limits(limits=1):
            partitions, configs = fit_clusterings(values)
        self.assertEqual(len(partitions), 9)
        self.assertEqual({c["method"] for c in configs}, {"K-Means", "Ward", "Leiden"})
        for variant, labels in partitions.items():
            with self.subTest(variant=variant):
                self.assertEqual(labels.shape, (90,))
                if "_k" in variant:
                    self.assertEqual(len(set(labels)), int(variant.split("_k")[1]))

    def test_silhouette_matches_sklearn_and_donor_means(self):
        from sklearn.metrics import silhouette_samples
        values = np.array([[0.0], [0.2], [2.0], [2.2], [2.4], [4.0]])
        cells = pd.DataFrame({"sample_id": ["a"] * 3 + ["b"] * 3})
        labels = np.array([0, 0, 1, 1, 1, 2])
        summary, donors, indices, fallback = silhouette_evaluation(
            values, cells, {"test": labels}, cells_per_donor=3)
        expected = silhouette_samples(values, labels)
        assert_allclose(summary.silhouette.iloc[0], (expected[:3].mean() + expected[3:].mean()) / 2)
        assert_allclose(donors.silhouette, [expected[:3].mean(), expected[3:].mean()])
        self.assertEqual(expected[-1], 0)
        self.assertFalse(fallback)
        assert_array_equal(indices, np.arange(6))

    def test_missing_cluster_switches_all_solutions_to_full_sample(self):
        values = np.arange(20).reshape(-1, 1)
        cells = pd.DataFrame({"sample_id": ["a"] * 10 + ["b"] * 10})
        regular = np.repeat([0, 1], 10)
        _, _, chosen, _ = silhouette_evaluation(values, cells, {"regular": regular}, 2)
        omitted = next(i for i in range(20) if i not in chosen)
        rare = regular.copy()
        rare[omitted] = 2
        summary, _, indices, fallback = silhouette_evaluation(
            values, cells, {"regular": regular, "rare": rare}, 2)
        self.assertTrue(fallback)
        self.assertTrue(summary.evaluation_cells.eq(20).all())
        assert_array_equal(indices, np.arange(20))
        invalid, _, _, _ = silhouette_evaluation(values, cells, {"one": np.zeros(20)}, 2)
        self.assertTrue(np.isnan(invalid.silhouette.iloc[0]))
        self.assertTrue(invalid.invalid_reason.iloc[0])

    def test_selection_tie_prefers_fewer_clusters_then_grid_order(self):
        summary = pd.DataFrame({"variant": ["a", "b", "c", "invalid"], "method": ["test"] * 4,
                                "n_clusters": [4, 2, 2, 1], "silhouette": [0.5, 0.5, 0.5, np.nan]})
        self.assertEqual(select_clusterings(summary), ["b"])

    def test_profiles_balance_present_donors_without_zero_imputation(self):
        cells = pd.DataFrame({"sample_id": ["a", "a", "a", "b", "b"]})
        values = pd.DataFrame({"marker": [0.0, 0.0, 9.0, 10.0, 12.0]})
        profiles, counts, _, donor_counts = cluster_profiles(values, cells, np.array([0, 0, 1, 0, 0]))
        self.assertEqual(profiles.loc[0, "marker"], 5.5)  # median(0, median(10,12))
        self.assertEqual(profiles.loc[1, "marker"], 9.0)  # absent b is not expression zero
        self.assertEqual(counts.loc[1, "donor_coverage"], 1)
        self.assertEqual(counts.loc[1, "dominant_donor_share"], 1)
        self.assertEqual(donor_counts.loc[1, "b"], 0)


if __name__ == "__main__":
    unittest.main()
