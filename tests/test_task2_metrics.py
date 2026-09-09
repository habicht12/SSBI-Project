"""Small deterministic checks; no real data or embedding training required."""

import unittest

import numpy as np

from src.task2_metrics import knn_preservation, neighbor_indices, neighborhood_overlap


class NeighborOverlapTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(42).normal(size=(40, 4))

    def test_identity_including_one_neighbor(self):
        for k in (1, 15, 39):
            self.assertEqual(knn_preservation(self.x, self.x, k), 1.0)

    def test_exact_count_and_self_exclusion_with_duplicates(self):
        x = np.vstack([self.x, self.x[:2]])
        indices = neighbor_indices(x, 15)
        self.assertEqual(indices.shape, (42, 15))
        self.assertTrue(all(i not in row for i, row in enumerate(indices)))

    def test_known_overlap(self):
        a = np.array([[1, 2], [0, 2], [0, 1], [0, 1]])
        b = np.array([[2, 3], [2, 3], [0, 3], [0, 2]])
        self.assertEqual(neighborhood_overlap(a, b), 0.5)

    def test_symmetry_bounds_and_rigid_transformation(self):
        y = np.random.default_rng(17).normal(size=(40, 2))
        ab = knn_preservation(self.x, y)
        self.assertEqual(ab, knn_preservation(y, self.x))
        self.assertTrue(0 <= ab <= 1)
        self.assertEqual(knn_preservation(self.x, -3 * self.x + 2), 1.0)

    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            knn_preservation(self.x, self.x[:-1])
        for k in (0, len(self.x)):
            with self.assertRaises(ValueError):
                neighbor_indices(self.x, k)


if __name__ == "__main__":
    unittest.main()
