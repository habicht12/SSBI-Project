"""OOF-Interpretation vorhandener Aufgabe-4-Modelle, ohne erneuten Modellfit."""

import hashlib
import json
import subprocess
import tempfile
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.task23_analysis import sample_event_indices


COEFFICIENT_TOLERANCE = 1e-10
METHODS = ("SVM", "CellCNN", "Citrus")
PROFILE_MARKERS = ["CD3", "CD4", "CD8", "CD19", "CD33", "CD11b", "CD56", "CD16",
                   "CD94", "NKG2A", "NKG2C", "CD57"]


def file_hash(path):
    """SHA256 einer Eingabedatei, ohne den gesamten Inhalt im RAM zu halten."""
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def csv_read(path):
    return pd.read_csv(path, float_precision="round_trip")


def align_projection(cells, embeddings):
    """Strenger Eins-zu-eins-Join; ID und Ereignis-/Spenderangaben müssen stimmen."""
    keys = ["cell_id", "sample_id", "event_index"]
    projection = embeddings.loc[embeddings.variant.eq("tsne_p30")]
    if cells.cell_id.duplicated().any() or projection.cell_id.duplicated().any():
        raise ValueError("Doppelte Zell-IDs in Projektionsdaten.")
    if set(cells.cell_id) != set(projection.cell_id):
        raise ValueError("Zellmengen der Projektion stimmen nicht überein.")
    result = cells.merge(projection[keys + ["component_1", "component_2"]],
                         on=keys, how="left", validate="one_to_one", sort=False)
    if result[["component_1", "component_2"]].isna().any().any():
        raise ValueError("Spender oder Originalereignis passt nicht zur Zell-ID.")
    if not np.isfinite(result[["component_1", "component_2"]]).all().all():
        raise ValueError("Nicht endliche Projektionskoordinaten.")
    return result


def load_inputs(root):
    """Originale Modellwerte float32; Projektion und biologische Profile float64."""
    import flowkit as fk

    root = Path(root)
    tables = root / "results/tables"
    data_root = root / "NK_cell_dataset/NK_cell_dataset"
    labels_path = data_root / "NK_fcs_samples_with_labels.csv"
    marker_path = data_root / "NK_markers.csv"
    marker_names = pd.read_csv(marker_path, header=None).iloc[0].dropna().tolist()
    cells = csv_read(tables / "task2_cells.csv")
    provenance = json.loads((tables / "task2_provenance.json").read_text())
    if len(marker_names) != 37 or len(set(marker_names)) != 37:
        raise ValueError("37 eindeutige Marker erforderlich.")
    if provenance["markers"] != marker_names or provenance["gate"] != "gated_alive":
        raise ValueError("Markerreihenfolge/Gate von Aufgabe 2 passt nicht.")
    if len(cells) != 10000 or not cells.groupby("sample_id").size().eq(500).all():
        raise ValueError("Die Karte muss 500 Zellen je Spender enthalten.")
    projection = align_projection(cells, csv_read(tables / "task2_embeddings.csv"))
    labels = csv_read(labels_path)
    donors = sorted(labels.fcs_filename.str.replace("_NK.fcs", "", regex=False))
    if donors != sorted(cells.sample_id.unique()) or len(donors) != 20:
        raise ValueError("Spenderliste der Daten und Karte stimmt nicht überein.")
    full_data, raw_parts, expected_parts = {}, [], []
    paths = [labels_path, marker_path, tables / "task4_donor_splits.csv"]
    for donor_index, donor in enumerate(donors):
        path = data_root / "NK_cell_dataset/gated_alive" / f"{donor}_alive.fcs"
        paths.append(path)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"FCS file .* reported incorrect data offset.*")
            sample = fk.Sample(str(path), ignore_offset_error=True)
        if len(set(sample.pns_labels)) != len(sample.pns_labels):
            raise ValueError(f"Doppelte Kanalnamen in {path.name}.")
        channels = [sample.pns_labels.index(marker) for marker in marker_names]
        raw = sample.get_events(source="raw")[:, channels]
        if not np.isfinite(raw).all():
            raise ValueError(f"Nicht endliche Markerwerte in {path.name}.")
        indices = sample_event_indices(len(raw), donor_index)
        expected_parts.append(pd.DataFrame({
            "cell_id": [f"gated_alive:{donor}:{i}" for i in indices],
            "sample_id": donor, "event_index": indices,
        }))
        raw_parts.append(raw[indices].astype(np.float64))
        # Genau wie 04b/04c: erst float32, dann arcsinh, NICHT umgekehrt.
        full_data[donor] = np.arcsinh(raw.astype(np.float32) / 5.0)
    expected = pd.concat(expected_parts, ignore_index=True)
    if not cells.equals(expected):
        raise ValueError("Sampling, Zellreihenfolge oder Originalereignisse wurden verändert.")
    raw_sample = np.concatenate(raw_parts)
    digest = hashlib.sha256()
    digest.update(json.dumps({"cell_ids": cells.cell_id.tolist(), "markers": marker_names},
                             separators=(",", ":")).encode())
    digest.update(raw_sample.astype("<f8").tobytes())
    if digest.hexdigest() != provenance["selected_data_sha256"]:
        raise ValueError("Rohdatenprüfsumme der Aufgabe-2-Karte stimmt nicht überein.")
    input_hashes = {path.name: file_hash(path) for path in paths}
    for method in ("svm", "cellcnn"):
        config = json.loads((tables / f"task4_{method}_predictions_gated_alive_full.config.json").read_text())
        if config["inputs"] != input_hashes:
            raise ValueError(f"Eingabedaten/Splits passen nicht zum {method}-Benchmark.")
        if config["parameters"]["gate"] != "gated_alive" or config["parameters"]["cofactor"] != 5:
            raise ValueError(f"Unpassende Transformation für {method}.")
    splits = csv_read(tables / "task4_donor_splits.csv")
    if splits.duplicated(["split_id", "donor_id"]).any() or set(splits.split_id) != set(range(100)):
        raise ValueError("Unvollständige oder doppelte Splits.")
    for _, group in splits.groupby("split_id"):
        if (set(group.donor_id) != set(donors) or len(group) != 20
                or group.outer_partition.eq("test").sum() != 6
                or group.outer_partition.eq("train").sum() != 14):
            raise ValueError("Ungültiger äußerer Spendersplit.")
    return dict(root=root, tables=tables, cells=projection, markers=marker_names,
                profiles=np.arcsinh(raw_sample / 5.0), data=full_data, splits=splits,
                input_hashes=input_hashes, selected_data_sha256=digest.hexdigest())


def restore_scaler(parameters):
    """Vorhandenen sklearn-Scaler rekonstruieren, insbesondere seine float32-Arithmetik."""
    scaler = StandardScaler()
    scaler.mean_ = parameters.scaler_mean.to_numpy(np.float64)
    scaler.scale_ = parameters.scaler_scale.to_numpy(np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(parameters)
    if not np.isfinite(scaler.mean_).all() or not np.all(scaler.scale_ > 0):
        raise ValueError("Ungültiger gespeicherter Scaler.")
    return scaler


def exact_top_indices(scores, fraction=0.01):
    """Genau ceil(fraction*n) größte Scores; Gleichstand nach Originalereignisindex."""
    scores = np.asarray(scores)
    if not len(scores) or not 0 < fraction <= 1 or not np.isfinite(scores).all():
        raise ValueError("Ungültige Zell-Scores oder Top-Fraktion.")
    count = max(1, int(np.ceil(fraction * len(scores))))
    boundary = np.partition(scores, -count)[-count]
    above = np.flatnonzero(scores > boundary)
    tied = np.flatnonzero(scores == boundary)[:count - len(above)]
    return np.sort(np.concatenate([above, tied]))


def svm_cell_selection(scores, threshold):
    """Richtung nur innerhalb der tatsächlich gepoolten höchsten 1 % bewerten."""
    indices = exact_top_indices(scores)
    selected = np.zeros(len(scores), dtype=bool)
    selected[indices] = True
    centered = scores - threshold
    return selected & (centered > 0), selected & (centered < 0), indices


def cellcnn_masks(responses, training_maxima, contrasts):
    """Papernahe Halbmaximum-Auswahl; gegensätzliche Filter können dieselbe Zelle wählen."""
    selected = (responses > 0.5 * training_maxima) & (training_maxima > 0)
    return (selected & (contrasts > 0)).any(axis=1), (selected & (contrasts < 0)).any(axis=1)


class SavedCellCNN(torch.nn.Module):
    """Nur Inferenz der gespeicherten Linear/ReLU/Top-Pooling/Linear-Architektur."""

    def __init__(self, filters, marker_names):
        super().__init__()
        filter_ids = sorted(filters.filter_id.unique())
        rows = [filters.loc[filters.filter_id.eq(i)].set_index("marker").loc[marker_names]
                for i in filter_ids]
        for row in rows:
            if len(row) != len(marker_names):
                raise ValueError("Unvollständiger CellCNN-Filter.")
        self.cell_filters = torch.nn.Linear(len(marker_names), len(rows))
        self.output_layer = torch.nn.Linear(len(rows), 2)
        with torch.no_grad():
            self.cell_filters.weight.copy_(torch.tensor(np.array([r.filter_weight for r in rows]), dtype=torch.float32))
            self.cell_filters.bias.copy_(torch.tensor([r.filter_bias.iloc[0] for r in rows], dtype=torch.float32))
            self.output_layer.weight.copy_(torch.tensor([
                [r.output_weight_0.iloc[0] for r in rows],
                [r.output_weight_1.iloc[0] for r in rows]], dtype=torch.float32))
            self.output_layer.bias.copy_(torch.tensor([
                rows[0].output_bias_0.iloc[0], rows[0].output_bias_1.iloc[0]], dtype=torch.float32))
        self.scaler = restore_scaler(rows[0])
        self.filter_ids = filter_ids
        self.eval()

    def forward(self, values):
        responses = torch.relu(self.cell_filters(values))
        count = max(1, int(0.01 * values.shape[0]))
        pooled = torch.topk(responses, k=count, dim=0).values.mean(dim=0)
        return self.output_layer(pooled), pooled


def validate_models(inputs, name, parameters, prediction_rows):
    """Split-Zuordnung und vollständige Markerparameter vor jeder Auswertung sichern."""
    expected = inputs["splits"].loc[inputs["splits"].outer_partition.eq("test")]
    keys = ["split_id", "donor_id"]
    if prediction_rows.duplicated(keys).any() or set(map(tuple, prediction_rows[keys].values)) != set(map(tuple, expected[keys].values)):
        raise ValueError(f"{name}: Vorhersagen passen nicht zu allen äußeren Testspendern.")
    joined = prediction_rows.merge(expected, on=keys, suffixes=("", "_split"), validate="one_to_one")
    if not joined.y_true.eq(joined.label).all() or not joined.split_seed.eq(joined.split_seed_split).all():
        raise ValueError(f"{name}: falsche Spenderlabels oder Splitseeds.")
    groups = ["split_id", "filter_id"] if name == "CellCNN" else ["split_id"]
    if parameters.duplicated(groups + ["marker"]).any() or set(parameters.split_id) != set(range(100)):
        raise ValueError(f"{name}: doppelte oder fehlende Modellparameter.")
    for _, group in parameters.groupby(groups):
        if set(group.marker) != set(inputs["markers"]):
            raise ValueError(f"{name}: Marker fehlen oder wurden ersetzt.")
    if (not parameters.gate.eq("gated_alive").all() or not parameters.run_mode.eq("full").all()
            or not parameters.cofactor.eq(5).all() or not parameters.top_fraction.eq(0.01).all()):
        raise ValueError(f"{name}: unpassende Modellkonfiguration.")


def run_python_methods(inputs, split_ids=range(100)):
    """Native SVM-/CellCNN-Vorhersagen prüfen, OOF-Selektion auf Projektionszellen exportieren."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    tables, cells, markers = inputs["tables"], inputs["cells"], inputs["markers"]
    svm = csv_read(tables / "task4_svm_models_gated_alive_full.csv")
    cnn = csv_read(tables / "task4_cellcnn_filters_gated_alive_full.csv")
    predictions = {method: csv_read(tables / f"task4_{name}_predictions_gated_alive_full.csv")
                   for method, name in [("SVM", "svm"), ("CellCNN", "cellcnn")]}
    validate_models(inputs, "SVM", svm, predictions["SVM"])
    validate_models(inputs, "CellCNN", cnn, predictions["CellCNN"])
    records, checks, thresholds = [], [], []
    for split_id in split_ids:
        split = inputs["splits"].loc[inputs["splits"].split_id.eq(split_id)]
        seed = int(split.split_seed.iloc[0])
        test_ids = sorted(split.loc[split.outer_partition.eq("test"), "donor_id"])
        svm_parameters = svm.loc[svm.split_id.eq(split_id)].set_index("marker").loc[markers]
        scaler = restore_scaler(svm_parameters)
        weights = svm_parameters.weight.to_numpy()
        intercept = float(svm_parameters.intercept.iloc[0])
        threshold = float(svm_parameters.decision_threshold.iloc[0])
        filters = cnn.loc[cnn.split_id.eq(split_id)]
        if filters.inner_fold.nunique() != 1:
            raise ValueError("Mehrere ausgewählte innere CellCNN-Folds.")
        train_ids = sorted(split.loc[split.outer_partition.eq("train") &
                                    split.inner_fold.ne(int(filters.inner_fold.iloc[0])), "donor_id"])
        if len(train_ids) not in (9, 10) or set(train_ids) & set(test_ids):
            raise ValueError("Falsche Trainingsspender für CellCNN-Referenzmaxima.")
        model = SavedCellCNN(filters, markers)
        device = torch.device("cuda" if torch.cuda.is_available() and filters.device.iloc[0] == "cuda" else "cpu")
        model.to(device)
        contrasts = (model.output_layer.weight[1] - model.output_layer.weight[0]).detach().cpu().numpy()
        maxima = np.zeros(len(model.filter_ids), dtype=np.float32)
        with torch.no_grad():
            for donor in train_ids:
                values = inputs["data"][donor]
                for start in range(0, len(values), 50000):
                    scaled = model.scaler.transform(values[start:start + 50000])
                    response = torch.relu(model.cell_filters(torch.from_numpy(scaled).to(device)))
                    maxima = np.maximum(maxima, response.max(dim=0).values.cpu().numpy())
        for fid, maximum, contrast in zip(model.filter_ids, maxima, contrasts):
            thresholds.append(dict(split_id=split_id, filter_id=fid, training_maximum=float(maximum),
                                   response_threshold=float(maximum / 2), contrast=float(contrast),
                                   training_donors=";".join(train_ids)))
        for donor_index, donor in enumerate(test_ids):
            subset = cells.loc[cells.sample_id.eq(donor)]
            indices = subset.event_index.to_numpy()
            values = inputs["data"][donor]
            margins = scaler.transform(values) @ weights + intercept
            pos, neg, top = svm_cell_selection(margins, threshold)
            score = float(margins[top].mean())
            expected = predictions["SVM"].loc[predictions["SVM"].split_id.eq(split_id) & predictions["SVM"].donor_id.eq(donor)].iloc[0]
            error = abs(score - expected.score)
            if error > 1e-10 or threshold != expected.decision_threshold or len(top) != expected.top_cell_count:
                raise ValueError(f"SVM-Rekonstruktion fehlgeschlagen: {split_id}/{donor}, Fehler {error}.")
            checks.append(dict(method="SVM", split_id=split_id, donor_id=donor,
                               saved_score=expected.score, reconstructed_score=score, score_error=error,
                               arithmetic_error=abs((score - threshold) - (margins[top] - threshold).mean()),
                               decision_matches=int(score >= threshold) == expected.y_pred))
            records.append(pd.DataFrame(dict(method="SVM", split_id=split_id, cell_id=subset.cell_id,
                                             positive=pos[indices], negative=neg[indices], ambiguous=False,
                                             exposed=True)))
            with torch.no_grad():
                responses = torch.relu(model.cell_filters(torch.from_numpy(model.scaler.transform(values[indices])).to(device)))
                pos, neg = cellcnn_masks(responses.cpu().numpy(), maxima, contrasts)
                rng = np.random.default_rng(seed + 900000 + donor_index)
                probabilities, arithmetic_errors = [], []
                exposed = np.zeros(len(indices), dtype=bool)
                for _ in range(5):
                    bag_indices = rng.choice(len(values), min(20000, len(values)), replace=False)
                    exposed |= np.isin(indices, bag_indices)
                    bag = torch.from_numpy(model.scaler.transform(values[bag_indices])).to(device)
                    logits, pooled = model(bag)
                    contrast_sum = ((model.output_layer.weight[1] - model.output_layer.weight[0]) * pooled).sum()
                    contrast_sum += model.output_layer.bias[1] - model.output_layer.bias[0]
                    arithmetic_errors.append(abs(float(logits[1] - logits[0] - contrast_sum)))
                    probabilities.append(float(torch.softmax(logits, dim=0)[1]))
                score = float(np.mean(probabilities))
            expected = predictions["CellCNN"].loc[predictions["CellCNN"].split_id.eq(split_id) & predictions["CellCNN"].donor_id.eq(donor)].iloc[0]
            error, arithmetic_error = abs(score - expected.score), max(arithmetic_errors)
            if (error > 1e-6 or arithmetic_error > 1e-5 or expected.prediction_inputs_per_donor != 5
                    or expected.prediction_cells_per_input != min(20000, len(values))):
                raise ValueError(f"CellCNN-Rekonstruktion fehlgeschlagen: {split_id}/{donor}, Fehler {error}/{arithmetic_error}.")
            checks.append(dict(method="CellCNN", split_id=split_id, donor_id=donor,
                               saved_score=expected.score, reconstructed_score=score, score_error=error,
                               arithmetic_error=arithmetic_error, decision_matches=int(score >= 0.5) == expected.y_pred))
            records.append(pd.DataFrame(dict(method="CellCNN", split_id=split_id, cell_id=subset.cell_id,
                                             positive=pos, negative=neg, ambiguous=pos & neg, exposed=exposed)))
        print(f"SVM/CellCNN: Split {split_id} rekonstruiert und geprüft.", flush=True)
    return pd.concat(records, ignore_index=True), pd.DataFrame(checks), pd.DataFrame(thresholds)


def run_citrus(inputs, split_ids=range(100)):
    """Nur finalen Trainingsbaum und Mapping im bestehenden R-Environment rekonstruieren."""
    with tempfile.TemporaryDirectory(prefix="task5_citrus_") as directory:
        command = ["conda", "run", "--no-capture-output", "-n", "ssbi-citrus", "Rscript",
                   str(inputs["root"] / "src/task5_citrus.R"), str(inputs["root"]), directory,
                   ",".join(map(str, split_ids))]
        subprocess.run(command, check=True)
        raw = csv_read(Path(directory) / "cells.csv")
        checks = csv_read(Path(directory) / "checks.csv")
        audit = csv_read(Path(directory) / "clusters.csv")
    raw["method"] = "Citrus"
    raw["positive"] = raw.raw_score > COEFFICIENT_TOLERANCE
    raw["negative"] = raw.raw_score < -COEFFICIENT_TOLERANCE
    raw["ambiguous"] = False
    raw["exposed"] = True  # Phänotypmapping ist für jede Projektionszelle definiert.
    return raw, checks, audit


def aggregate_frequencies(records, cells, splits):
    """Pro Zelle durch sämtliche zulässigen Testmodelle ihres Spenders teilen."""
    columns = ["method", "split_id", "cell_id"]
    if records.duplicated(columns).any():
        raise ValueError("Doppelte OOF-Zellbewertung.")
    joined = records.merge(cells[["cell_id", "sample_id"]], on="cell_id", how="left", validate="many_to_one")
    allowed = splits.loc[splits.outer_partition.eq("test"), ["split_id", "donor_id"]]
    validation = joined.merge(allowed, left_on=["split_id", "sample_id"], right_on=["split_id", "donor_id"],
                              how="left", validate="many_to_one")
    if validation.donor_id.isna().any():
        raise ValueError("Eine Zellbewertung verwendet einen Nicht-Testspender.")
    denominators = allowed.groupby("donor_id").size()
    expected_count = len(cells) * 3
    counts = joined.groupby(["method", "cell_id", "sample_id"], sort=False).agg(
        n_test_models=("split_id", "size"), positive_count=("positive", "sum"),
        negative_count=("negative", "sum"), ambiguous_count=("ambiguous", "sum"),
        exposed_models=("exposed", "sum")).reset_index()
    if len(counts) != expected_count or set(counts.method) != set(METHODS):
        raise ValueError("Nicht alle Methoden und Projektionszellen wurden bewertet.")
    if not counts.n_test_models.eq(counts.sample_id.map(denominators)).all():
        raise ValueError("OOF-Modelle fehlen; fehlende Daten sind keine Nullmodelle.")
    for kind in ("positive", "negative", "ambiguous"):
        counts[f"{kind}_frequency"] = counts[f"{kind}_count"] / counts.n_test_models
    return counts.merge(cells, on=["cell_id", "sample_id"], validate="many_to_one")


def weighted_median(values, weights):
    """Kleinster Wert mit kumuliertem positivem Gewicht >= der Hälfte; leer ergibt NaN."""
    values, weights = np.asarray(values), np.asarray(weights)
    if values.shape != weights.shape or not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Ungültige Daten/Gewichte für den gewichteten Median.")
    valid = weights > 0
    if not valid.any():
        return np.nan
    order = np.argsort(values[valid], kind="stable")
    values, weights = values[valid][order], weights[valid][order]
    index = np.searchsorted(np.cumsum(weights), weights.sum() / 2, side="left")
    return float(values[index])


def summarize_profiles(frequencies, cells, values, markers):
    """Profile der Kartenstichprobe: frequenzgewichtete Spender-, dann Spendermediane."""
    value_frame = pd.DataFrame(values, index=cells.cell_id, columns=markers)
    donor_profiles, donor_counts = [], []
    for method in METHODS:
        frame = frequencies.loc[frequencies.method.eq(method)]
        for direction in ("positive", "negative"):
            for donor, subset in frame.groupby("sample_id"):
                weights = subset[f"{direction}_frequency"].to_numpy()
                donor_values = value_frame.loc[subset.cell_id].to_numpy()
                donor_counts.append(dict(method=method, direction=direction, donor_id=donor,
                                         selected_cells=int((weights > 0).sum()), weight_sum=float(weights.sum()),
                                         selection_fraction=float(weights.mean())))
                for marker_index, marker in enumerate(markers):
                    donor_profiles.append(dict(method=method, direction=direction, donor_id=donor,
                                               marker=marker, value=weighted_median(donor_values[:, marker_index], weights)))
    for donor, subset in cells.groupby("sample_id"):
        donor_values = value_frame.loc[subset.cell_id].to_numpy()
        for index, marker in enumerate(markers):
            donor_profiles.append(dict(method="Kartenreferenz", direction="all", donor_id=donor,
                                       marker=marker, value=float(np.median(donor_values[:, index]))))
    donor_profiles, donor_counts = pd.DataFrame(donor_profiles), pd.DataFrame(donor_counts)
    profiles = donor_profiles.groupby(["method", "direction", "marker"], sort=False).agg(
        value=("value", "median"), supported_donors=("value", "count")).reset_index()
    counts = donor_counts.groupby(["method", "direction"], sort=False).agg(
        selected_cells=("selected_cells", "sum"), supported_donors=("weight_sum", lambda x: int((x > 0).sum())),
        mean_selection_fraction=("selection_fraction", "mean"),
        minimum_donor_cells=("selected_cells", "min"), maximum_donor_cells=("selected_cells", "max")).reset_index()
    return profiles, counts, donor_profiles, donor_counts


def run_interpretation(root):
    """Vollständiger Interpretationslauf; Aufgabe-4-Dateien werden ausschließlich gelesen."""
    inputs = load_inputs(root)
    python_records, python_checks, thresholds = run_python_methods(inputs)
    citrus_records, citrus_checks, citrus_audit = run_citrus(inputs)
    records = pd.concat([python_records, citrus_records], ignore_index=True)
    frequencies = aggregate_frequencies(records, inputs["cells"], inputs["splits"])
    checks = pd.concat([python_checks, citrus_checks], ignore_index=True)
    if len(checks) != 1800 or not checks.decision_matches.all():
        raise ValueError("Nicht alle 1.800 Modell-/Spendervorhersagen erfolgreich geprüft.")
    profiles, counts, donor_profiles, donor_counts = summarize_profiles(
        frequencies, inputs["cells"], inputs["profiles"], inputs["markers"])
    outputs = dict(cell_scores=frequencies, prediction_checks=checks, filter_thresholds=thresholds,
                   citrus_cluster_checks=citrus_audit, marker_profiles=profiles, selection_summary=counts,
                   donor_marker_profiles=donor_profiles, donor_selection_summary=donor_counts)
    for name, table in outputs.items():
        table.to_csv(inputs["tables"] / f"task5_{name}.csv", index=False)
    source_inputs = list(inputs["tables"].glob("task4_*gated_alive_full*"))
    source_inputs += [inputs["tables"] / f"task2_{name}" for name in ("cells.csv", "embeddings.csv", "provenance.json")]
    provenance = dict(
        gate="gated_alive", interpretation="outer-test-only selection frequencies; no classifier fitting; Citrus final training trees reconstructed",
        markers=inputs["markers"], selected_data_sha256=inputs["selected_data_sha256"],
        input_sha256=inputs["input_hashes"], artifact_sha256={p.name: file_hash(p) for p in source_inputs},
        implementation_sha256={name: file_hash(inputs["root"] / "src" / name)
                               for name in ("task5_interpretation.py", "task5_citrus.R")},
        package_versions={name: version(name) for name in ("numpy", "pandas", "scikit-learn", "torch", "flowkit")},
        projection="existing task2 tsne_p30, exact cell-ID join, no refit",
        score_data="float32 raw -> arcsinh(x/5) float32 -> original training StandardScaler",
        profile_data="float64 raw -> arcsinh(x/5), frequency-weighted within donor; median across supported donors",
        cutoff=dict(svm="exact top ceil(1% all donor cells), then sign(margin - donor threshold); ties event index",
                    cellcnn="response > 0.5 * max across all actual inner-training donor events; output contrast sign",
                    citrus="sum beta*overlapping membership; beta and net direction tolerance 1e-10"),
        reference_seed=42, oof_denominator="all outer test models for this donor, including null models",
        prediction_check_tolerance=dict(svm=1e-10, cellcnn_probability=1e-6,
                                        cellcnn_logit_identity=1e-5, citrus_centered_logit=1e-8),
        output_sha256={f"task5_{name}.csv": file_hash(inputs["tables"] / f"task5_{name}.csv") for name in outputs},
    )
    (inputs["tables"] / "task5_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return outputs
