"""Kleine Rechen- und Datenzuordnungstests ohne Training oder Originaldaten."""

import unittest

import numpy as np
import pandas as pd
import torch
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.preprocessing import StandardScaler

from src.task5_interpretation import (
    METHODS, SavedCellCNN, aggregate_frequencies, align_projection,
    cellcnn_masks, exact_top_indices, restore_scaler, summarize_profiles,
    svm_cell_selection, weighted_median,
)


class InterpretationTests(unittest.TestCase):
    def test_top_selection_ties_exact_count_and_threshold_direction(self):
        scores = np.zeros(101)
        scores[:4] = [9, 8, 8, 8]
        assert_array_equal(exact_top_indices(scores), [0, 1])
        positive, negative, top = svm_cell_selection(scores, 8.5)
        assert_array_equal(np.flatnonzero(positive), [0])
        assert_array_equal(np.flatnonzero(negative), [1])
        self.assertAlmostEqual((scores[top] - 8.5).mean(), scores[top].mean() - 8.5)
        self.assertFalse(negative[-1])  # Kleinste Margins werden nicht gepoolt.

    def test_scaler_preserves_original_float32_rounding(self):
        values = np.array([[0.000123, 340.001], [5.017, 28.131], [8.22, 88.19]], dtype=np.float32)
        original = StandardScaler().fit(values)
        restored = restore_scaler(pd.DataFrame(dict(scaler_mean=original.mean_, scaler_scale=original.scale_)))
        assert_array_equal(original.transform(values), restored.transform(values))
        self.assertEqual(restored.transform(values).dtype, np.float32)

    def test_cellcnn_halfmax_zero_and_conflicting_filters(self):
        responses = np.array([[9, 8, 3, 0], [5, 5, 1, 0], [0, 9, 2, 0]])
        positive, negative = cellcnn_masks(responses, np.array([10, 10, 3, 0]), np.array([1, -1, 0, 1]))
        assert_array_equal(positive, [True, False, False])
        assert_array_equal(negative, [True, False, True])

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

    def test_projection_joins_identity_and_rejects_wrong_donor(self):
        cells = pd.DataFrame(dict(cell_id=["a", "b"], sample_id=["d", "d"], event_index=[1, 2]))
        embeddings = cells.iloc[::-1].assign(variant="tsne_p30", component_1=[3., 4.], component_2=0.)
        assert_array_equal(align_projection(cells, embeddings).component_1, [4, 3])
        embeddings.loc[0, "sample_id"] = "wrong"
        with self.assertRaises(ValueError):
            align_projection(cells, embeddings)
        with self.assertRaises(ValueError):
            align_projection(cells, pd.concat([embeddings, embeddings]))

    def test_oof_denominator_keeps_null_models_and_rejects_training(self):
        cells = pd.DataFrame(dict(cell_id=["a", "b"], sample_id=["d1", "d2"], event_index=[1, 2]))
        splits = pd.DataFrame(dict(split_id=[0, 1, 0, 1], donor_id=["d1", "d1", "d2", "d2"],
                                   outer_partition=["test", "test", "test", "train"]))
        records = pd.DataFrame([dict(method=m, split_id=s, cell_id=c, positive=(s == 0),
                                     negative=False, ambiguous=False, exposed=True)
                                for m in METHODS for s, c in [(0, "a"), (1, "a"), (0, "b")]])
        result = aggregate_frequencies(records, cells, splits)
        assert_array_equal(result.loc[result.cell_id.eq("a"), "positive_frequency"], [0.5] * 3)
        assert_array_equal(result.loc[result.cell_id.eq("b"), "positive_frequency"], [1] * 3)
        with self.assertRaises(ValueError):
            aggregate_frequencies(records.iloc[1:], cells, splits)
        invalid = records.copy()
        invalid.loc[2, "split_id"] = 1
        with self.assertRaises(ValueError):
            aggregate_frequencies(invalid, cells, splits)

    def test_weighted_median_lower_tie_zero_and_invalid_weights(self):
        self.assertEqual(weighted_median([9, 1, 5], [1, 1, 0]), 1)
        self.assertEqual(weighted_median([9, 1, 5], [2, 1, 1]), 5)
        self.assertTrue(np.isnan(weighted_median([1, 2], [0, 0])))
        with self.assertRaises(ValueError):
            weighted_median([1, 2], [1, -1])

    def test_profiles_weight_donors_equally_without_zero_imputation(self):
        cells = pd.DataFrame(dict(cell_id=["a", "b", "c", "d"], sample_id=["x", "x", "y", "z"]))
        frequencies = pd.concat([cells.assign(method=m, positive_frequency=[1., 1., 1., 0.], negative_frequency=0.)
                                  for m in METHODS], ignore_index=True)
        profiles, counts, _, _ = summarize_profiles(frequencies, cells, np.array([[2], [4], [20], [99]]), ["m"])
        positive = profiles.loc[profiles.direction.eq("positive")]
        assert_allclose(positive.value, [11, 11, 11])  # lower donor x median=2; y=20
        assert_array_equal(positive.supported_donors, [2, 2, 2])
        self.assertTrue(profiles.loc[profiles.direction.eq("negative"), "value"].isna().all())
        assert_array_equal(counts.loc[counts.direction.eq("positive"), "selected_cells"], [3, 3, 3])


if __name__ == "__main__":
    unittest.main()
