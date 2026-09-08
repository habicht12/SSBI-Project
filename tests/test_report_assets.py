"""Small checks against silent cell misalignment in report exports."""
import unittest
from pathlib import Path
import tempfile

import pandas as pd

from src.report_assets import align_cells, metric_table


class ReportCellAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.cells = pd.DataFrame({"cell_id": ["a", "b"],
                                   "sample_id": ["donor1", "donor2"],
                                   "event_index": [0, 1]})

    def test_reordering_preserves_measurement_identity(self):
        measurements = self.cells.assign(score=[.2, .8]).iloc[::-1]
        aligned = align_cells(measurements, self.cells)
        self.assertEqual(aligned.score.tolist(), [.2, .8])

    def test_rejects_missing_duplicate_or_misattributed_cells(self):
        invalid = [self.cells.iloc[:1],
                   pd.concat([self.cells.iloc[:1], self.cells.iloc[:1]]),
                   self.cells.assign(sample_id=["wrong", "donor2"]),
                   self.cells.assign(event_index=[3, 1])]
        for frame in invalid:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                align_cells(frame, self.cells)


class ClassificationTableTests(unittest.TestCase):
    def write_tables(self, directory, pairwise=False, count=None):
        count = count if count is not None else (100 if pairwise else 10)
        methods = ["CellCNN", "Lineare Single-Cell-SVM"]
        if not pairwise:
            methods.append("Citrus")
        values = {"roc_auc": .75, "average_precision": .6, "balanced_accuracy": .625}
        records = [dict(method_label=method, split_id=split, **values)
                   for method in methods for split in range(count)]
        summary = [dict(Methode=method, Metrik=metric, Median=value, Q1=value, Q3=value)
                   for method in methods for metric, value in values.items()]
        prefix = "task4_cellcnn_svm_100" if pairwise else "task4"
        pd.DataFrame(records).to_csv(directory / f"{prefix}_split_metrics.csv", index=False)
        pd.DataFrame(summary).to_csv(directory / f"{prefix}_metric_summary.csv", index=False)

    def test_distinct_comparison_scopes(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.write_tables(directory)
            self.write_tables(directory, pairwise=True)
            three_way = metric_table(directory)
            pair = metric_table(directory, pairwise=True)
            self.assertIn("10 gemeinsame Splits", three_way)
            self.assertIn("Citrus", three_way)
            self.assertIn("100 gemeinsame Splits", pair)
            self.assertNotIn("Citrus", pair)
            self.assertIn("0.750 [0.750, 0.750]", pair)

    def test_rejects_wrong_scope_or_stale_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.write_tables(directory, count=100)
            with self.assertRaisesRegex(ValueError, "10 common"):
                metric_table(directory)
            self.write_tables(directory)
            path = directory / "task4_metric_summary.csv"
            summary = pd.read_csv(path)
            summary.loc[0, "Median"] = .5
            summary.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                metric_table(directory)


if __name__ == "__main__":
    unittest.main()
