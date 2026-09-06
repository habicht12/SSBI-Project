"""Konfiguration und Vollständigkeit lokaler Aufgabe-4-Zwischenstände prüfen."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_digest(path):
    """Datei ausschließlich lesen und ihren Inhalt mit SHA-256 kennzeichnen."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_run_config(parameters, input_paths, notebook_path, code_cells):
    """Parameter, Eingangsdaten und relevante Notebook-Zellen festhalten.

    Split-Limits und erzwungene Wiederholungen ändern das Modellverfahren nicht
    und gehören nicht in parameters. Die gesamte Splitdatei wird dagegen gehasht.
    """
    notebook = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    source = "\n\n".join(
        "".join(notebook["cells"][index]["source"]) for index in code_cells
    )
    return {
        "schema_version": 1,
        "parameters": parameters,
        "inputs": {Path(path).name: file_digest(path) for path in input_paths},
        "implementation_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


def check_run_config(config_path, expected, artifact_paths, run_training):
    """Bei unvollständigen, alten oder anders konfigurierten Artefakten stoppen.

    Schreibt nichts. Ein neuer Konfigurationsnachweis wird erst zusammen mit
    dem ersten erfolgreich berechneten Split gespeichert.
    """
    config_path = Path(config_path)
    paths = [Path(path) for path in artifact_paths]
    present = [path.exists() for path in paths]
    if not config_path.exists() and not any(present):
        if not run_training:
            raise FileNotFoundError("Keine Ergebnisse vorhanden; zuerst einen Lauf berechnen.")
        return False
    if not config_path.exists():
        raise ValueError(
            f"Konfigurationsnachweis fehlt: {config_path.name}. "
            "Altbestände können in 04e historisch ausgewertet werden. "
            "Für einen neuen Methodenlauf die bisherigen Artefakte separat sichern "
            "und aus den aktiven Ergebnispfaden verschieben; sie werden nicht überschrieben."
        )
    stored = json.loads(config_path.read_text(encoding="utf-8"))
    if stored != expected:
        changed = sorted(key for key in set(stored) | set(expected)
                         if stored.get(key) != expected.get(key))
        raise ValueError(
            "Gespeicherte Ergebnisse passen nicht zur aktuellen Konfiguration "
            f"({', '.join(changed)}). Alte Artefakte separat sichern und "
            "für die neue Konfiguration einen getrennten Lauf beginnen."
        )
    if not all(present):
        missing = ", ".join(path.name for path in paths if not path.exists())
        raise ValueError(f"Unvollständiger Zwischenstand; fehlende Artefakte: {missing}")
    return True


def write_run_config(path, config):
    """Den tatsächlich verwendeten Konfigurationsnachweis speichern."""
    Path(path).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_prediction_splits(predictions, donor_splits):
    """Nur vollständig gespeicherte äußere Spendersplits zum Fortsetzen zulassen."""
    columns = ["split_id", "split_seed", "donor_id", "y_true"]
    if predictions.empty or predictions.duplicated(["split_id", "donor_id"]).any():
        raise ValueError("Leerer oder doppelt gespeicherter Vorhersage-Zwischenstand.")
    expected = donor_splits.loc[
        donor_splits["outer_partition"].eq("test")
        & donor_splits["split_id"].isin(predictions["split_id"]),
        ["split_id", "split_seed", "donor_id", "label"],
    ].rename(columns={"label": "y_true"})
    matched = predictions[columns].merge(expected, on=columns, validate="one_to_one")
    if len(matched) != len(predictions) or len(matched) != len(expected):
        raise ValueError("Gespeicherte Vorhersagen sind unvollständig oder passen nicht zu den Splits.")
    if not np.isfinite(predictions[["score", "decision_threshold"]].to_numpy()).all():
        raise ValueError("Nicht-endliche Scores oder Entscheidungsschwellen.")
    if not predictions["y_pred"].eq(
        predictions["score"].ge(predictions["decision_threshold"]).astype(int)
    ).all():
        raise ValueError("Gespeicherte Klassen stimmen nicht mit Score und Schwelle überein.")


def validate_parameter_table(parameters, predictions, markers, group_columns, required_columns=()):
    """Pro Modell/Filter genau einen endlichen Parametersatz je Marker verlangen."""
    missing = set(required_columns) - set(parameters.columns)
    if missing:
        raise ValueError(f"Fehlende Modellparameterspalten: {', '.join(sorted(missing))}")
    if required_columns and not np.isfinite(parameters[list(required_columns)].to_numpy(dtype=float)).all():
        raise ValueError("Nicht-endliche gespeicherte Modellparameter.")
    if set(parameters["split_id"]) != set(predictions["split_id"]):
        raise ValueError("Modellparameter und Vorhersagen enthalten unterschiedliche Splits.")
    if parameters.duplicated([*group_columns, "marker"]).any():
        raise ValueError("Doppelte Marker in gespeicherten Modellparametern.")
    for _, group in parameters.groupby(group_columns):
        if len(group) != len(markers) or set(group["marker"]) != set(markers):
            raise ValueError("Unvollständige Marker in gespeicherten Modellparametern.")
    if not np.isfinite(parameters.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Nicht-endliche gespeicherte Modellparameter.")
