"""Small checks against silent cell misalignment in report exports."""
import unittest

import pandas as pd

from src.report_assets import align_cells


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


if __name__ == "__main__":
    unittest.main()
