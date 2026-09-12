"""Kleine unabhängige Tests der Auswahl, Zentroidgruppen und Datenzuordnung."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.preprocessing import StandardScaler

from src.task5_interpretation import (
    CENTROID_COLUMNS, SPLIT_IDS, SavedCellCNN, aggregate_svm_frequencies, align_projection,
    citrus_centroids, exact_top_indices, filter_responses, group_centroids,
    halfmax_centroids, project_centroids, representative_cells, restore_scaler,
    run_interpretation, svm_cell_selection, training_reference,
)


class InterpretationTests(unittest.TestCase):
    def setUp(self):
        self.markers = ["m1", "m2"]
        self.exploration = dict(scaler_mean=[0., 0.], scaler_scale=[1., 1.])

    def centroids(self, rows):
        return pd.DataFrame([dict(method=m, split_id=s, subset_id=f, coefficient=a,
                                  m1=x, m2=y) for m, s, f, a, x, y in rows])

    def test_top_selection_ties_exact_count_and_threshold_direction(self):
        scores = np.zeros(101)
        scores[:4] = [9, 8, 8, 8]
        assert_array_equal(exact_top_indices(scores), [0, 1])
        positive, negative, top = svm_cell_selection(scores, 8.5)
        assert_array_equal(np.flatnonzero(positive), [0])
        assert_array_equal(np.flatnonzero(negative), [1])
        self.assertAlmostEqual((scores[top] - 8.5).mean(), scores[top].mean() - 8.5)
        self.assertFalse(negative[-1])

    def test_cellcnn_native_pooling_floor_and_logit_identity(self):
        rows = []
        for fid, alpha in [(0, 2.), (1, -3.)]:
            rows.append(dict(filter_id=fid, marker="m", filter_weight=1., filter_bias=0.,
                             output_weight_0=0., output_weight_1=alpha,
                             output_bias_0=0.2, output_bias_1=-0.4, scaler_mean=0., scaler_scale=1.))
        model = SavedCellCNN(pd.DataFrame(rows), ["m"])
        logits, pooled = model(torch.arange(101, dtype=torch.float32)[:, None])
        assert_array_equal(pooled.detach().numpy(), [100, 100])  # floor=1, SVM ceil=2
        self.assertAlmostEqual(float((logits[1] - logits[0]).detach()), -100.6, places=4)
        bag_logits = torch.tensor([[-1., 1.], [-1., 5.]])
        self.assertNotAlmostEqual(float(torch.softmax(bag_logits, 1)[:, 1].mean()),
                                  float(torch.sigmoid((bag_logits[:, 1] - bag_logits[:, 0]).mean())))

    def test_scaler_preserves_original_float32_rounding(self):
        values = np.array([[.000123, 340.001], [5.017, 28.131], [8.22, 88.19]], dtype=np.float32)
        original = StandardScaler().fit(values)
        restored = restore_scaler(pd.DataFrame(dict(scaler_mean=original.mean_, scaler_scale=original.scale_)))
        assert_array_equal(original.transform(values), restored.transform(values))
        self.assertEqual(restored.transform(values).dtype, np.float32)

    def test_native_filter_response_uses_saved_scaler_relu_and_sorted_ids(self):
        filters = pd.DataFrame([
            dict(filter_id=f, marker=m, filter_weight=w, filter_bias=b, scaler_mean=1., scaler_scale=2.)
            for f, b, weights in [(2, -1., [1., 1.]), (0, 0., [2., -1.])]
            for m, w in zip(self.markers, weights)])
        response = filter_responses(np.array([[3, 5], [5, 3]], dtype=np.float32), filters, self.markers)
        assert_array_equal(response, [[0, 2], [3, 2]])

    def test_halfmax_strict_boundary_centroid_and_null_filters(self):
        values = np.array([[2., 10.], [4., 30.], [8., 50.]])
        responses = np.array([[10., 0., 2.], [5., 0., 9.], [8., 0., 8.]])
        result = halfmax_centroids(values, responses, np.array([-2., 1., 0.]))
        self.assertEqual(len(result), 1)
        index, threshold, count, centroid = result[0]
        self.assertEqual((index, threshold, count), (0, 5., 2))
        assert_array_equal(centroid, [5, 30])

    def test_reference_uses_actual_inner_training_donors_and_original_seed(self):
        split = pd.DataFrame(dict(donor_id=[f"d{i:02}" for i in range(20)], split_seed=23,
                                   outer_partition=["train"] * 14 + ["test"] * 6,
                                   inner_fold=[i % 3 for i in range(14)] + [-1] * 6))
        selected = pd.Series(dict(inner_fold=1, filter_count=3, candidate_seed=10323))
        expected_ids = [f"d{i:02}" for i in range(14) if i % 3 != 1]
        # Validierungs- und Testspender fehlen absichtlich im Datenwörterbuch.
        data = {d: np.full((20000, 2), i, dtype=np.float32) for i, d in enumerate(expected_ids)}
        reference, donors = training_reference(data, split, selected)
        self.assertEqual(donors, expected_ids)
        self.assertEqual(reference.shape, (180000, 2))
        assert_array_equal(np.bincount(reference[:, 0].astype(int)), [20000] * 9)
        selected.candidate_seed += 1
        with self.assertRaisesRegex(ValueError, "Kandidatenseed"):
            training_reference(data, split, selected)

    def test_groups_count_splits_once_keep_null_splits_and_mixed_directions(self):
        rows = [("CellCNN", s, 0, 1., 1., 0.) for s in range(6)]
        rows += [("CellCNN", 0, 1, -1., 1., 0.), ("CellCNN", 6, 0, 1., -1., 0.)]
        rows += [("Citrus", 0, 0, 1., 1., 0.)]
        centroids, groups = group_centroids(self.centroids(rows), self.markers, self.exploration, SPLIT_IDS)
        group = groups.loc[groups.method.eq("CellCNN") & groups.group_id.eq(1)].iloc[0]
        self.assertEqual((group.n_centroids, group.occurrences, group.n_splits), (7, 6, 30))
        self.assertAlmostEqual(group.frequency, .2)
        self.assertTrue(group.retained)
        self.assertEqual((group.positive_splits, group.negative_splits), (6, 1))
        self.assertEqual((group.representative_split_id, group.representative_subset_id), (0, 0))
        self.assertEqual(groups.retained.sum(), 1)
        self.assertEqual(len(centroids), 9)

    def test_fixed_cutoff_and_medoid_are_deterministic_under_row_reordering(self):
        # Winkel 0, 30, 60 Grad: Average-Linkage verbindet alle bei Schnitt 0,4.
        angles = np.deg2rad([0, 30, 60])
        frame = self.centroids([("CellCNN", i, 0, 1., np.cos(a), np.sin(a)) for i, a in enumerate(angles)])
        _, first = group_centroids(frame, self.markers, self.exploration, SPLIT_IDS)
        _, second = group_centroids(frame.iloc[::-1], self.markers, self.exploration, SPLIT_IDS)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 1)
        self.assertEqual(first.representative_split_id.iloc[0], 1)
        self.assertFalse(first.retained.iloc[0])

    def test_empty_singleton_invalid_cosine_and_duplicate_centroids(self):
        empty = pd.DataFrame(columns=CENTROID_COLUMNS + self.markers)
        result, groups = group_centroids(empty, self.markers, self.exploration, SPLIT_IDS)
        self.assertTrue(result.empty and groups.empty)
        singleton = self.centroids([("Citrus", 0, 1, 1., 1., 0.)])
        _, groups = group_centroids(singleton, self.markers, self.exploration, SPLIT_IDS)
        self.assertEqual(groups.occurrences.iloc[0], 1)
        for invalid in [singleton.assign(m1=0), singleton.assign(m1=np.nan), pd.concat([singleton, singleton])]:
            with self.assertRaises(ValueError):
                group_centroids(invalid, self.markers, self.exploration, SPLIT_IDS)

    def test_projection_uses_euclidean_scaled_marker_space_and_exact_cell_ids(self):
        cells = pd.DataFrame(dict(cell_id=["a", "b"], component_1=[11., 22.], component_2=[3., 4.]))
        frame = self.centroids([("Citrus", 0, 1, 1., 3., 0.)])
        # Im Rohraum wäre b näher, nach gespeicherter Skalierung a.
        result = project_centroids(frame, cells, [[0., 0.], [3., 2.]], self.markers,
                                   dict(scaler_mean=[0., 0.], scaler_scale=[100., 1.]))
        self.assertEqual(result.map_cell_id.iloc[0], "a")
        self.assertEqual(result.component_1.iloc[0], 11.)
        empty = project_centroids(frame.iloc[:0], cells, [[0., 0.], [3., 2.]], self.markers, self.exploration)
        self.assertTrue(empty.empty)

    def test_projection_join_rejects_wrong_donor_or_duplicate_identity(self):
        cells = pd.DataFrame(dict(cell_id=["a", "b"], sample_id=["d", "d"], event_index=[1, 2]))
        embeddings = cells.iloc[::-1].assign(variant="tsne_p30", component_1=[3., 4.], component_2=0.)
        assert_array_equal(align_projection(cells, embeddings).component_1, [4, 3])
        embeddings.loc[0, "sample_id"] = "wrong"
        with self.assertRaises(ValueError):
            align_projection(cells, embeddings)
        with self.assertRaises(ValueError):
            align_projection(cells, pd.concat([embeddings, embeddings]))

    def test_svm_denominators_missing_models_and_training_leakage(self):
        cells = pd.DataFrame(dict(cell_id=["a", "b", "c"], sample_id=["d1", "d2", "d3"]))
        splits = pd.DataFrame(dict(split_id=[0, 1, 0, 1], donor_id=["d1", "d1", "d2", "d2"],
                                   outer_partition=["test", "test", "test", "train"]))
        records = pd.DataFrame(dict(split_id=[0, 1, 0], cell_id=["a", "a", "b"],
                                    positive=[True, False, True], negative=False))
        result = aggregate_svm_frequencies(records, cells, splits)
        assert_allclose(result.positive_frequency, [.5, 1., np.nan])
        for invalid in [records.iloc[1:], pd.concat([records, records.iloc[:1]]),
                        records.assign(split_id=[0, 1, 1]), records.assign(cell_id=["unknown", "a", "b"])]:
            with self.assertRaises(ValueError):
                aggregate_svm_frequencies(invalid, cells, splits)

    def test_citrus_saved_centroids_and_null_models(self):
        profiles = pd.DataFrame(dict(split_id=[0, 0], cluster_id=5, marker=self.markers,
                                     coefficient=-2., centroid=[3., 7.], gate="gated_alive",
                                     transform_cofactor=5, minimum_cluster_size_fraction=.0005))
        selection = pd.DataFrame(dict(split_id=[0, 1], selected_cluster_count=[1, 0],
                                      file_sample_size=10000, minimum_cluster_size_fraction=.0005))
        artifacts = dict(citrus_clusters=profiles, citrus_selection=selection, split_ids=(0, 1))
        result = citrus_centroids(artifacts, self.markers)
        assert_allclose(result[self.markers], [[3, 7]])
        self.assertEqual(result.coefficient.iloc[0], -2.)
        with self.assertRaisesRegex(ValueError, "Clusterzahl"):
            citrus_centroids(artifacts | dict(citrus_clusters=profiles.iloc[:0]), self.markers)
        with self.assertRaisesRegex(ValueError, "Konfiguration"):
            citrus_centroids(artifacts | dict(citrus_selection=selection.assign(file_sample_size=1000)), self.markers)
        null = artifacts | dict(citrus_clusters=profiles.iloc[:0], citrus_selection=selection.assign(selected_cluster_count=0))
        self.assertTrue(citrus_centroids(null, self.markers).empty)

    def test_no_representative_when_no_group_meets_six_split_threshold(self):
        frame = self.centroids([("CellCNN", 0, 0, 1., 1., 0.)])
        centroids, groups = group_centroids(frame, self.markers, self.exploration, SPLIT_IDS)
        result = representative_cells(dict(cells=pd.DataFrame(columns=["cell_id"])), {}, centroids, groups)
        self.assertTrue(result.empty)
        self.assertIn("selected", result)

    def test_full_run_checks_completeness_before_reading_fcs_or_exporting(self):
        with patch("src.task5_interpretation.load_artifacts", side_effect=ValueError("Splits fehlen")), \
             patch("src.task5_interpretation.load_inputs") as loader:
            with self.assertRaisesRegex(ValueError, "Splits fehlen"):
                run_interpretation("unused")
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
