"""Die Blockberechnung muss die Referenz auch bei Bindungen reproduzieren."""
import unittest
import numpy as np
from sklearn.manifold import trustworthiness
from src.task2_metrics import trustworthiness_chunked

class TrustworthinessTests(unittest.TestCase):
    def test_matches_sklearn_across_dtypes_and_chunk_boundaries(self):
        rng = np.random.default_rng(42)
        for dtype in [np.float32, np.float64]:
            x = rng.normal(size=(43, 7)).astype(dtype)
            y = rng.normal(size=(43, 2)).astype(dtype)
            for k in [1, 5, 15]:
                expected = trustworthiness(x, y, n_neighbors=k)
                for size in [1, 8, 128]:
                    self.assertAlmostEqual(trustworthiness_chunked(x, y, k, size), expected, places=14)

    def test_duplicates_and_equal_distances(self):
        x = np.array([[0,0], [0,0], [1,0], [-1,0], [0,1], [0,-1], [2,0], [-2,0]], dtype=float)
        y = x[:, ::-1]
        self.assertAlmostEqual(trustworthiness_chunked(x,y,2,3), trustworthiness(x,y,n_neighbors=2), places=14)

    def test_invalid_alignment_and_neighborhood(self):
        x = np.arange(24).reshape(8,3)
        for y,k,size in [(x[:7],2,3),(x,4,3),(x,0,3),(x,2,0)]:
            with self.assertRaises(ValueError):trustworthiness_chunked(x,y,k,size)

if __name__ == '__main__':unittest.main()
