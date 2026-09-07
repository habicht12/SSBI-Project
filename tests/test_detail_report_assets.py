"""Handrechnungen für den vom Benchmark unabhängigen Berichtsabgleich."""
import unittest

import numpy as np

from src.detail_report_assets import independent_metrics


class IndependentMetricTests(unittest.TestCase):
    def test_constant_scores_distinguish_ap_from_trapezoidal_area(self):
        result = independent_metrics([0, 0, 0, 0, 1, 1], [.5] * 6, [1] * 6)
        self.assertAlmostEqual(result["roc_auc"], .5)
        self.assertAlmostEqual(result["average_precision"], 1 / 3)
        self.assertAlmostEqual(result["pr_auc"], 2 / 3)
        self.assertAlmostEqual(result["balanced_accuracy"], .5)

    def test_ties_are_grouped_and_roc_pairs_receive_half_credit(self):
        result = independent_metrics([1, 0, 1, 0], [3, 3, 1, 0], [1, 1, 0, 0])
        self.assertAlmostEqual(result["roc_auc"], .625)
        self.assertAlmostEqual(result["average_precision"], .5 * .5 + .5 * 2 / 3)
        self.assertAlmostEqual(result["pr_auc"], .5 * .75 + .5 * (0.5 + 2 / 3) / 2)

    def test_perfect_ranking_and_invalid_inputs(self):
        result = independent_metrics([0, 0, 1, 1], [0, 1, 2, 3], [0, 0, 1, 1])
        self.assertTrue(all(value == 1 for value in result.values()))
        for scores in [[0, np.nan], [1], [[1, 2]]]:
            with self.assertRaises(ValueError):
                independent_metrics([0, 1], scores, [0, 1])


if __name__ == "__main__":
    unittest.main()
