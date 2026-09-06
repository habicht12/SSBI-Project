"""Regressionstests für Konfigurationskonflikte und unvollständige Zwischenstände."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.task4_artifacts import (
    check_run_config, make_run_config, validate_parameter_table,
    validate_prediction_splits, write_run_config,
)


class ArtifactTests(unittest.TestCase):
    def test_changed_config_or_inputs_never_reuse_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "splits.csv"
            data.write_text("split_id,inner_fold\n0,1\n")
            notebook = root / "model.ipynb"
            notebook.write_text(json.dumps({"cells": [{"source": ["model(C=C)"]}]}))
            def config(c_values):
                return make_run_config({"C_values": c_values}, [data], notebook, [0])
            original = config([0.01, 0.1, 1.0])
            manifest = root / "config.json"
            artifacts = [root / "predictions.csv", root / "models.csv"]
            self.assertFalse(check_run_config(manifest, original, artifacts, True))
            with self.assertRaises(FileNotFoundError):
                check_run_config(manifest, original, artifacts, False)
            for path in artifacts:
                path.write_text("existing results\n")
            with self.assertRaisesRegex(ValueError, "Konfigurationsnachweis fehlt"):
                check_run_config(manifest, original, artifacts, True)
            write_run_config(manifest, original)
            for training in [False, True]:
                self.assertTrue(check_run_config(manifest, original, artifacts, training))
                with self.assertRaisesRegex(ValueError, "Konfiguration"):
                    check_run_config(manifest, config([1.0]), artifacts, training)
            # Same donor names and seed, but a changed inner fold must be detected.
            data.write_text("split_id,inner_fold\n0,2\n")
            with self.assertRaisesRegex(ValueError, "inputs"):
                check_run_config(manifest, config([0.01, 0.1, 1.0]), artifacts, True)
            data.write_text("split_id,inner_fold\n0,1\n")
            notebook.write_text(json.dumps({"cells": [{"source": ["different_model(C=C)"]}]}))
            with self.assertRaisesRegex(ValueError, "implementation_sha256"):
                check_run_config(manifest, config([0.01, 0.1, 1.0]), artifacts, True)
            artifacts[1].unlink()
            with self.assertRaisesRegex(ValueError, "Unvollständiger Zwischenstand"):
                check_run_config(manifest, original, artifacts, True)

    def test_incomplete_or_misassigned_donor_predictions(self):
        splits = pd.DataFrame({
            "split_id": [0, 0], "split_seed": [7, 7], "donor_id": ["a", "b"],
            "label": [0, 1], "outer_partition": ["test", "test"],
        })
        predictions = splits.drop(columns="outer_partition").rename(columns={"label": "y_true"})
        predictions = predictions.assign(score=[0.2, 0.8], decision_threshold=0.5, y_pred=[0, 1])
        validate_prediction_splits(predictions, splits)
        for invalid in [
            predictions.iloc[:1],
            predictions.assign(split_seed=8),
            predictions.assign(y_true=[1, 0]),
            predictions.assign(y_pred=[1, 0]),
            predictions.assign(score=float("nan")),
        ]:
            with self.subTest(invalid=invalid.to_dict()):
                with self.assertRaises(ValueError):
                    validate_prediction_splits(invalid, splits)

    def test_missing_filter_or_marker_parameters(self):
        predictions = pd.DataFrame({"split_id": [0, 1]})
        parameters = pd.DataFrame({
            "split_id": [0, 0, 1, 1], "marker": ["CD3", "CD56"] * 2,
            "weight": [0.1, 0.2, 0.3, 0.4],
        })
        validate_parameter_table(parameters, predictions, ["CD3", "CD56"], ["split_id"])
        with self.assertRaisesRegex(ValueError, "Modellparameterspalten"):
            validate_parameter_table(parameters.drop(columns="weight"), predictions,
                                     ["CD3", "CD56"], ["split_id"], ["weight"])
        for invalid in [parameters.iloc[:-1], parameters.iloc[:2],
                        pd.concat([parameters, parameters.iloc[:1]]),
                        parameters.assign(weight=float("nan"))]:
            with self.assertRaises(ValueError):
                validate_parameter_table(invalid, predictions, ["CD3", "CD56"], ["split_id"])


if __name__ == "__main__":
    unittest.main()
