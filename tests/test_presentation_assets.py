"""Small independent checks for the SVM profile added to the presentation."""

import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from src.presentation_assets import aggregate_svm_profiles, svm_profile_records


class SVMProfileTests(unittest.TestCase):
    def test_equal_donors_and_equal_nonempty_visits_not_cell_counts(self):
        records = pd.DataFrame(dict(
            split_id=[0, 1, 2, 0, 0], donor_id=["a", "a", "a", "b", "c"],
            n_selected=[100, 1, 0, 2, 0], m1=[0, 10, np.nan, 20, np.nan],
            m2=[2, 6, np.nan, 12, np.nan]))
        profile, coverage = aggregate_svm_profiles(records, ("m1", "m2"))
        assert_allclose(profile, [12.5, 8])  # a: [5, 4], b: [20, 12]; c has no profile.
        self.assertEqual(coverage, dict(test_visits=5, nonempty_test_visits=3,
                                       test_donors=3, represented_donors=2,
                                       selected_cell_occurrences=103))

    def test_empty_and_invalid_profiles_fail(self):
        empty = pd.DataFrame(dict(split_id=[0], donor_id=["a"], n_selected=[0], m1=[np.nan]))
        with self.assertRaisesRegex(ValueError, "keine positiv"):
            aggregate_svm_profiles(empty, ("m1",))
        valid = empty.assign(n_selected=1, m1=2.)
        for invalid in (pd.concat([valid, valid]), valid.assign(n_selected=-1), valid.assign(m1=np.nan),
                        pd.concat([valid, empty.assign(split_id=1, m1=0.)])):
            with self.subTest(invalid=invalid.to_dict()), self.assertRaises(ValueError):
                aggregate_svm_profiles(invalid, ("m1",))

    def fixture(self, threshold):
        values = np.zeros((101, 2), dtype=np.float32)
        values[:4] = [[9, 30], [8, 100], [8, 900], [8, 1000]]
        cells = pd.DataFrame(dict(cell_id=["c0", "c1", "c2", "c3"], sample_id="test",
                                  event_index=range(4)))
        inputs = dict(markers=["m1", "m2"], cells=cells, data={"test": values})
        models = pd.DataFrame(dict(split_id=0, marker=["m1", "m2"], weight=[1., 0.],
                                   intercept=0., scaler_mean=0., scaler_scale=1., decision_threshold=threshold))
        splits = pd.DataFrame(dict(split_id=[0, 0], donor_id=["train", "test"],
                                   outer_partition=["train", "test"]))
        predictions = pd.DataFrame(dict(split_id=[0], donor_id=["test"], score=[8.5],
                                        decision_threshold=[threshold], top_cell_count=[2], y_true=[0]))
        saved = cells.assign(positive_count=[1, 0, 0, 0], negative_count=[0, int(threshold > 8), 0, 0],
                             n_test_models=1)
        return inputs, models, predictions, splits, saved

    def test_full_sample_top_ties_and_strict_threshold_on_negative_donor(self):
        for threshold in (8., 8.5):
            with self.subTest(threshold=threshold):
                records = svm_profile_records(*self.fixture(threshold), markers=("m1", "m2"))
                # ceil(1% of 101) = 2, not one from the four map cells. Training donor has no input.
                self.assertEqual(records.n_top.tolist(), [2])
                self.assertEqual(records.n_selected.tolist(), [1])
                assert_allclose(records[["m1", "m2"]], [[9., 30.]])

    def test_score_and_map_count_mismatches_fail(self):
        inputs, models, predictions, splits, saved = self.fixture(8.5)
        with self.assertRaisesRegex(ValueError, "prediction mismatch"):
            svm_profile_records(inputs, models, predictions.assign(score=8.4), splits, saved, ("m1", "m2"))
        with self.assertRaises(AssertionError):
            svm_profile_records(inputs, models, predictions, splits, saved.assign(positive_count=0), ("m1", "m2"))

    def test_empty_selected_subset_is_missing_and_counted(self):
        inputs, models, predictions, splits, saved = self.fixture(10.)
        saved = saved.assign(positive_count=0, negative_count=[1, 1, 0, 0])
        records = svm_profile_records(inputs, models, predictions, splits, saved, ("m1", "m2"))
        self.assertEqual(records.n_selected.tolist(), [0])
        self.assertTrue(records[["m1", "m2"]].isna().all().all())


if __name__ == "__main__":
    unittest.main()
