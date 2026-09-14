"""Independent checks of exact Citrus training-event mapping."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal
from scipy.spatial import cKDTree

from src.task5_citrus_oof import exact_nearest, exact_positive_membership


class CitrusMappingTests(unittest.TestCase):
    def test_mapping_matches_exhaustive_euclidean_search(self):
        rng = np.random.default_rng(4526)
        reference = rng.normal(size=(173, 37))
        query = rng.normal(size=(83, 37))
        expected = ((query[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        assert_array_equal(exact_nearest(cKDTree(reference), reference, query, workers=1), expected)

    def test_ties_choose_first_original_training_event(self):
        reference = np.array([[1., 0.], [-1., 0.], [1., 0.], [0., 10.]])
        query = np.array([[0., 0.], [1., 0.], [0., 9.]])
        assert_array_equal(exact_nearest(cKDTree(reference), reference, query, workers=1), [0, 0, 3])

    def test_membership_is_inherited_from_training_cell_not_cluster_centroid(self):
        reference = np.array([[0.], [100.], [4.]])
        # Positive cluster members: 0 and 100, centroid=50. The query at 0.1
        # belongs to it via the nearest training cell despite the distant mean.
        positive = np.array([True, True, False])
        nearest = exact_nearest(cKDTree(reference), reference, [[.1], [4.1]], workers=1)
        assert_array_equal(positive[nearest], [True, False])

    def test_invalid_inputs_fail(self):
        reference = np.array([[0.], [1.]])
        with self.assertRaises(ValueError):
            exact_nearest(cKDTree(reference), reference, [[np.nan]])

    def test_negative_witness_acceleration_preserves_exact_membership(self):
        rng = np.random.default_rng(4567)
        reference = rng.normal(size=(2000, 37))
        query = np.concatenate([rng.normal(size=(500, 37)), reference[:10]])
        tree = cKDTree(reference)
        nearest = ((query[:, None] - reference[None]) ** 2).sum(axis=2).argmin(axis=1)
        for fraction in [0., .001, .1, .5, 1.]:
            positive = rng.random(len(reference)) < fraction
            assert_array_equal(exact_positive_membership(tree, reference, positive, query, workers=1), positive[nearest])

    def test_negative_witness_does_not_break_cross_class_ties(self):
        reference = np.array([[1., 0.], [-1., 0.], [1., 0.]])
        tree = cKDTree(reference)
        for positive in [np.array([True, False, False]), np.array([False, True, True])]:
            assert_array_equal(exact_positive_membership(tree, reference, positive, np.array([[0., 0.], [1., 0.]]), workers=1), [positive[0], positive[0]])


if __name__ == "__main__":
    unittest.main()
