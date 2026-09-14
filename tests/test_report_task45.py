"""Check statistical units and selection edge cases for the additional figures."""

import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from src.report_task45 import positive_filter_union, profile_summaries, group_profile_summaries, performance_tables
from src.task5_interpretation import aggregate_svm_frequencies


class ReportTests(unittest.TestCase):
    def test_union_uses_positive_filters_strict_training_threshold_and_single_vote(self):
        # Two positive filters can select the same cell; negative and zero-max
        # filters cannot select even when their response on a new cell is high.
        responses = [[8, 9, 100, 100], [5, 4, 100, 100], [0, 5, 100, 100]]
        result = positive_filter_union(responses, [5, 4, 1, 0], [1, 2, -3, 1])
        assert_array_equal(result, [True, False, True])
        self.assertEqual(result.dtype, bool)
        assert_array_equal(positive_filter_union(responses, [5, 4, 1, 0], [-1, -2, -3, 0]), [False]*3)
        with self.assertRaises(ValueError):
            positive_filter_union([[np.nan]], [1], [1])

    def test_cell_recurrence_counts_empty_visits_and_rejects_training_donors(self):
        cells = pd.DataFrame(dict(cell_id=["a", "b"], sample_id=["d1", "d2"]))
        splits = pd.DataFrame(dict(split_id=[0, 1, 2, 0, 1], donor_id=["d1"]*3 + ["d2"]*2,
                                   outer_partition=["test"]*4 + ["train"]))
        records = pd.DataFrame(dict(split_id=[0, 1, 2, 0], cell_id=["a"]*3 + ["b"],
                                    positive=[True, False, False, True], negative=False))
        result = aggregate_svm_frequencies(records, cells, splits)
        assert_allclose(result.positive_frequency, [1/3, 1])
        with self.assertRaises(ValueError):
            aggregate_svm_frequencies(records.iloc[:-1], cells, splits)
        bad = records.copy()
        bad.loc[3, "split_id"] = 1
        with self.assertRaises(ValueError):
            aggregate_svm_frequencies(bad, cells, splits)

    def test_profiles_weight_donors_equally_and_keep_empty_selection_undefined(self):
        visits = pd.DataFrame(dict(method="CellCNN", donor_id=["a", "a", "a", "b", "c"],
                                   n_selected=[100, 1, 0, 500, 0], m=[2., 6., np.nan, 10., np.nan]))
        donors, summary = profile_summaries(visits, ["m"], [0], [1])
        self.assertEqual(summary.mean_z.iloc[0], 7.)  # donor a = 4; b = 10
        self.assertEqual(summary.n_donors.iloc[0], 2)
        self.assertEqual(donors.loc[donors.donor_id.eq("a"), "test_visits"].iloc[0], 3)
        self.assertTrue(np.isnan(donors.loc[donors.donor_id.eq("c"), "m"].iloc[0]))

    def test_group_profiles_do_not_overweight_splits_with_multiple_filters(self):
        centroids = pd.DataFrame(dict(method="CellCNN", group_id=1, split_id=[0, 0, 1], m=[0., 2., 9.]))
        groups = pd.DataFrame(dict(method=["CellCNN"], group_id=[1], retained=[True]))
        splits, summary = group_profile_summaries(centroids, groups, ["m"], [0], [1])
        assert_array_equal(splits.m, [1, 9])
        self.assertEqual(summary.median_z.iloc[0], 5)
        self.assertEqual(summary.n_splits.iloc[0], 2)

    def test_paired_differences_align_split_ids_even_with_shuffled_predictions(self):
        artifacts = {}
        for method in ["svm", "cellcnn", "citrus"]:
            rows = []
            for split_id in [1, 0]:
                # SVM ranks perfectly on split 0 and reverses all pairs on 1.
                score = [0, 0, 0, 0, 1, 1] if method != "svm" or split_id == 0 else [1, 1, 1, 1, 0, 0]
                rows.extend(dict(split_id=split_id, y_true=y, score=s, y_pred=int(s >= .5))
                            for y, s in zip([0, 0, 0, 0, 1, 1], score))
            artifacts[f"{method}_predictions"] = pd.DataFrame(rows).sample(frac=1, random_state=4)
        _, _, differences, summary = performance_tables(artifacts)
        pair = differences.loc[differences["first"].eq("CellCNN") & differences.second.eq("SVM")]
        assert_array_equal(pair.sort_values("split_id").auc_difference, [0, 1])
        row = summary.loc[summary["first"].eq("CellCNN") & summary.second.eq("SVM")].iloc[0]
        self.assertEqual((row.better, row.equal, row.worse), (1, 1, 0))


if __name__ == "__main__":
    unittest.main()
