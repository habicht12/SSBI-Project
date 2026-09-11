import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.export_task6 import OUT, load_data, frequency_values, donor_frequencies

class Task6ExportTests(unittest.TestCase):
    def test_frozen_benchmark_counts_and_means(self):
        data, common, _ = load_data(OUT / "data")
        self.assertEqual(len(common), 98)
        self.assertEqual(set(range(100)) - common, {44, 84})
        summary = data["summary"].set_index("model")
        self.assertEqual(summary.loc["quadratic_top1", "network_auc_mean"], .9125)
        self.assertEqual(summary.loc["quadratic_soft", "network_auc_mean"], .9125)
        wide = data["split_metrics"].pivot(index="split_id", columns="model", values="network_auc")
        delta = wide.quadratic_soft - wide.quadratic_top1
        self.assertEqual((int(delta.gt(0).sum()), int(delta.eq(0).sum()), int(delta.lt(0).sum())), (5, 90, 5))
        self.assertEqual(data["learned_alphas"].groupby("model").size().to_dict(),
                         {"prototype_soft": 414, "quadratic_soft": 416})

    def test_frequency_scope_percentage_points_and_direction(self):
        frame = pd.DataFrame({"model": ["a"]*3, "split_id": [0, 1, 2],
            "frequency_effect": [.01, -.02, .99], "frequency_auc": [.25, .75, 1.]})
        np.testing.assert_array_equal(frequency_values(frame, "a", "frequency_effect", {0, 1}), [1., -2.])
        np.testing.assert_array_equal(frequency_values(frame, "a", "frequency_auc", {0, 1}), [.25, .75])

    def test_donors_are_not_weighted_by_number_of_test_visits(self):
        frame = pd.DataFrame({"donor_id": ["a", "a", "a", "b"], "y_true": [0, 0, 0, 1],
                              "frequency": [0., 0., 0., 1.]})
        points = donor_frequencies(frame)
        self.assertEqual(len(points), 2)
        self.assertEqual(points.frequency.mean(), .5)

    def test_modified_source_file_is_rejected(self):
        source = OUT / "data"
        manifest = json.loads((source / "input_provenance.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name in ["input_provenance.json", *manifest["input_sha256"]]:
                shutil.copy2(source / name, target / name)
            path = target / "frequencies.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Benchmark input changed"):
                load_data(target)

if __name__ == "__main__":
    unittest.main()
