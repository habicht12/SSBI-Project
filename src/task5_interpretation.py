"""Papernahe Subset-Zentroiden und SVM-OOF-Auswahl aus gespeicherten Modellen."""

import hashlib
import json
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform, cdist

from src.task4_artifacts import (
    validate_comparison_configs, validate_prediction_splits, validate_parameter_table,
)

from src.task23_analysis import sample_event_indices


COEFFICIENT_TOLERANCE = 1e-10
METHODS = ("SVM", "CellCNN", "Citrus")
SPLIT_IDS = tuple(range(30))
REFERENCE_CELLS_PER_DONOR = 20000
CENTROID_COLUMNS = ["method", "split_id", "subset_id", "coefficient", "response_threshold",
                    "selected_cells", "reference_cells", "candidate_seed", "training_donors"]
GROUP_COLUMNS = ["method", "group_id", "n_centroids", "n_splits", "occurrences", "frequency",
                 "positive_splits", "negative_splits", "representative_split_id",
                 "representative_subset_id", "retained"]


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
                input_hashes=input_hashes, selected_data_sha256=digest.hexdigest(),
                exploration=provenance)


def restore_scaler(parameters):
    """Vorhandenen sklearn-Scaler rekonstruieren, insbesondere seine float32-Arithmetik."""
    scaler = StandardScaler()
    scaler.mean_ = parameters.scaler_mean.to_numpy(np.float64)
    scaler.scale_ = parameters.scaler_scale.to_numpy(np.float64)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(parameters)
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all() or not np.all(scaler.scale_ > 0):
        raise ValueError("Ungültiger gespeicherter Scaler.")
    return scaler


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


def load_artifacts(root, split_ids=SPLIT_IDS):
    """Explizite Splits prüfen; Teilmengen dienen nur exportfreien technischen Tests."""
    root = Path(root)
    split_ids = tuple(split_ids)
    if not split_ids or len(set(split_ids)) != len(split_ids) or not set(split_ids) <= set(SPLIT_IDS):
        raise ValueError("Erwartet werden eindeutige Split-IDs aus 0–29.")
    tables = root / "results/tables"
    splits = csv_read(tables / "task4_donor_splits.csv")
    splits = splits.loc[splits.split_id.isin(split_ids)]
    markers = pd.read_csv(root / "NK_cell_dataset/NK_cell_dataset/NK_markers.csv",
                          header=None).iloc[0].dropna().tolist()
    artifacts = dict(split_ids=split_ids, splits=splits)
    paths = [tables / "task4_donor_splits.csv"]
    for method in ("svm", "cellcnn", "citrus"):
        for kind in ("predictions", "selection", {"svm": "models", "cellcnn": "filters", "citrus": "clusters"}[method]):
            path = tables / f"task4_{method}_{kind}_gated_alive_full.csv"
            frame = csv_read(path)
            if kind != "clusters":
                missing = sorted(set(split_ids) - set(frame.split_id))
                if missing:
                    raise ValueError(f"{method}/{kind}: Splits {missing} fehlen. "
                                     "Aufgabe 5 benötigt vollständig gespeicherte Splits 0–29; "
                                     "den laufenden Citrus-Lauf zuerst fertigrechnen lassen.")
            artifacts[f"{method}_{kind}"] = frame.loc[frame.split_id.isin(split_ids)].copy()
            paths.append(path)
        extension = "rds" if method == "citrus" else "json"
        paths.append(tables / f"task4_{method}_predictions_gated_alive_full.config.{extension}")
        validate_prediction_splits(artifacts[f"{method}_predictions"], splits)
    validate_comparison_configs(root)
    for method, kind, groups in [("svm", "models", ["split_id"]),
                                  ("cellcnn", "filters", ["split_id", "filter_id"])]:
        parameters = artifacts[f"{method}_{kind}"]
        validate_parameter_table(parameters, artifacts[f"{method}_predictions"], markers, groups)
        if (not parameters.gate.eq("gated_alive").all() or not parameters.run_mode.eq("full").all()
                or not parameters.cofactor.eq(5).all() or not parameters.top_fraction.eq(.01).all()):
            raise ValueError(f"{method}: falsche Transformation oder Modellkonfiguration.")
    selected = artifacts["cellcnn_selection"].loc[artifacts["cellcnn_selection"].selected]
    if selected.split_id.duplicated().any() or set(selected.split_id) != set(split_ids):
        raise ValueError("Genau ein ausgewähltes CellCNN-Modell je Split erforderlich.")
    artifacts["cellcnn_selection"] = selected
    artifacts["source_hashes"] = {p.name: file_hash(p) for p in paths}
    return artifacts


def training_reference(data, split, selection):
    """Dieselbe spenderbalancierte Ziehung wie fit_balanced_scaler in Aufgabe 4."""
    train_ids = sorted(split.loc[split.outer_partition.eq("train") &
                                split.inner_fold.ne(selection.inner_fold), "donor_id"])
    expected_seed = int(split.split_seed.iloc[0]) + 10000 * int(selection.inner_fold) + 100 * int(selection.filter_count)
    if (len(train_ids) not in (9, 10) or selection.candidate_seed != expected_seed
            or set(train_ids) & set(split.loc[split.outer_partition.eq("test"), "donor_id"])):
        raise ValueError("CellCNN: Trainingsspender oder Kandidatenseed stimmen nicht.")
    rng = np.random.default_rng(int(selection.candidate_seed))
    parts = []
    for donor in train_ids:
        values = data[donor]
        if len(values) < REFERENCE_CELLS_PER_DONOR:
            raise ValueError(f"Zu wenige Trainingszellen: {donor}.")
        indices = rng.choice(len(values), size=REFERENCE_CELLS_PER_DONOR, replace=False)
        parts.append(values[indices])
    return np.concatenate(parts), train_ids


def filter_responses(values, filters, markers):
    """Gespeicherte Filter auf der CPU auswerten; float32 wie beim Modelltraining."""
    rows = [group.set_index("marker").loc[markers]
            for _, group in filters.groupby("filter_id", sort=True)]
    scaler = restore_scaler(rows[0])
    weights = torch.tensor(np.array([r.filter_weight for r in rows]), dtype=torch.float32)
    biases = torch.tensor([r.filter_bias.iloc[0] for r in rows], dtype=torch.float32)
    with torch.no_grad():
        values = torch.from_numpy(scaler.transform(values))
        return torch.relu(torch.nn.functional.linear(values, weights, biases)).numpy()


def halfmax_centroids(values, responses, contrasts):
    """Ungewichteter Subset-Mittelwert nach spenderbalancierter Referenzziehung."""
    if (len(values) != len(responses) or responses.shape[1] != len(contrasts)
            or not all(np.isfinite(x).all() for x in (values, responses, contrasts))):
        raise ValueError("Ungültige Referenzwerte oder Filterantworten.")
    maxima = responses.max(axis=0)
    result = []
    for index, contrast in enumerate(contrasts):
        if maxima[index] <= 0 or contrast == 0:
            continue
        selected = responses[:, index] > .5 * maxima[index]
        result.append((index, .5 * float(maxima[index]), int(selected.sum()),
                       values[selected].mean(axis=0, dtype=np.float64)))
    return result


def cellcnn_centroids(inputs, artifacts):
    """Ein Zentroid je wirksamem Filter; keine Vereinigung verschiedener Filter."""
    records = []
    for selected in artifacts["cellcnn_selection"].sort_values("split_id").itertuples():
        split = artifacts["splits"].loc[artifacts["splits"].split_id.eq(selected.split_id)]
        filters = artifacts["cellcnn_filters"].loc[artifacts["cellcnn_filters"].split_id.eq(selected.split_id)]
        meta = filters.groupby("filter_id", sort=True).first()
        if len(meta) != selected.filter_count or not filters.inner_fold.eq(selected.inner_fold).all():
            raise ValueError("CellCNN-Filter passen nicht zum ausgewählten Kandidaten.")
        reference, donors = training_reference(inputs["data"], split, selected)
        responses = filter_responses(reference, filters, inputs["markers"])
        contrasts = (meta.output_weight_1.to_numpy(np.float32) - meta.output_weight_0.to_numpy(np.float32))
        for index, threshold, count, centroid in halfmax_centroids(reference, responses, contrasts):
            records.append(dict(method="CellCNN", split_id=selected.split_id,
                                subset_id=int(meta.index[index]), coefficient=float(contrasts[index]),
                                response_threshold=threshold, selected_cells=count,
                                reference_cells=len(reference), candidate_seed=int(selected.candidate_seed),
                                training_donors=";".join(donors), **dict(zip(inputs["markers"], centroid))))
    return pd.DataFrame(records, columns=CENTROID_COLUMNS + inputs["markers"])


def citrus_centroids(artifacts, markers):
    """Vorhandene Trainingszentroiden lesen; numerische Nullmodelle nicht verlieren."""
    profiles, selection = artifacts["citrus_clusters"], artifacts["citrus_selection"]
    if selection.split_id.duplicated().any() or set(selection.split_id) != set(artifacts["split_ids"]):
        raise ValueError("Citrus: unvollständige oder doppelte Modellauswahl.")
    if (not selection.file_sample_size.eq(10000).all()
            or not selection.minimum_cluster_size_fraction.eq(.0005).all()
            or profiles.duplicated(["split_id", "cluster_id", "marker"]).any()
            or not profiles.gate.eq("gated_alive").all()
            or not profiles.transform_cofactor.eq(5).all()
            or not profiles.minimum_cluster_size_fraction.eq(.0005).all()):
        raise ValueError("Citrus: alte Konfiguration oder doppelte Clusterparameter.")
    if not np.isfinite(profiles[["coefficient", "centroid"]]).all().all():
        raise ValueError("Citrus: nicht endliche Clusterparameter.")
    records = []
    for (split_id, cluster_id), group in profiles.groupby(["split_id", "cluster_id"], sort=True):
        if set(group.marker) != set(markers) or len(group) != len(markers) or group.coefficient.nunique() != 1:
            raise ValueError("Citrus: unvollständiger oder widersprüchlicher Zentroid.")
        coefficient = float(group.coefficient.iloc[0])
        if abs(coefficient) > COEFFICIENT_TOLERANCE:
            records.append(dict(method="Citrus", split_id=split_id, subset_id=cluster_id,
                                coefficient=coefficient,
                                **group.set_index("marker").centroid.loc[markers].to_dict()))
    result = pd.DataFrame(records, columns=CENTROID_COLUMNS + markers)
    counts = result.groupby("split_id").size()
    if not selection.selected_cluster_count.eq(selection.split_id.map(counts).fillna(0)).all():
        raise ValueError("Citrus: Clusterzahl passt nicht zur gespeicherten Auswahl.")
    return result


def exploratory_values(values, exploration):
    """Gemeinsamer gespeicherter Darstellungsraum; kein neuer Scalerfit."""
    mean = np.asarray(exploration["scaler_mean"])
    scale = np.asarray(exploration["scaler_scale"])
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Ungültige explorative Skalierung.")
    return (np.asarray(values, dtype=np.float64) - mean) / scale


def group_centroids(centroids, markers, exploration, split_ids):
    """Average/Kosinus/0,4; Wiederkehr zählt Splits, nicht Filter oder Cluster."""
    result = centroids.sort_values(["method", "split_id", "subset_id"]).reset_index(drop=True).copy()
    if result.duplicated(["method", "split_id", "subset_id"]).any() or not set(result.split_id) <= set(split_ids):
        raise ValueError("Doppelte Zentroiden oder unzulässige Splits.")
    result["group_id"] = pd.Series(index=result.index, dtype="int64")
    summaries = []
    for method, frame in result.groupby("method", sort=True):
        values = exploratory_values(frame[markers], exploration)
        if not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any():
            raise ValueError("Kosinusdistanz für nicht endliche oder Null-Zentroiden undefiniert.")
        distances = np.clip(pdist(values, metric="cosine"), 0, 2)
        labels = fcluster(linkage(distances, method="average"), .4, criterion="distance") if len(frame) > 1 else np.ones(1, dtype=int)
        groups = []
        for label in np.unique(labels):
            members = frame.loc[labels == label]
            local = values[labels == label]
            totals = squareform(np.clip(pdist(local, metric="cosine"), 0, 2)).sum(axis=1)
            representative = members.iloc[np.flatnonzero(np.isclose(totals, totals.min(), rtol=0, atol=1e-12))[0]]
            occurrences = members.split_id.nunique()
            groups.append((members.index, dict(
                method=method, n_centroids=len(members), n_splits=len(split_ids),
                occurrences=occurrences, frequency=occurrences / len(split_ids),
                positive_splits=members.loc[members.coefficient.gt(0), "split_id"].nunique(),
                negative_splits=members.loc[members.coefficient.lt(0), "split_id"].nunique(),
                representative_split_id=int(representative.split_id),
                representative_subset_id=int(representative.subset_id), retained=occurrences >= 6)))
        groups.sort(key=lambda item: (-item[1]["occurrences"], item[1]["representative_split_id"], item[1]["representative_subset_id"]))
        for group_id, (indices, summary) in enumerate(groups, start=1):
            result.loc[indices, "group_id"] = group_id
            summaries.append(dict(group_id=group_id, **summary))
    result["group_id"] = result.group_id.astype(int)
    return result, pd.DataFrame(summaries, columns=GROUP_COLUMNS)


def project_centroids(centroids, cells, profiles, markers, exploration):
    """Nächste echte Karten-Zelle im euklidischen z-Raum; Gleichstand nach Kartenreihenfolge."""
    result = centroids.copy()
    values = exploratory_values(result[markers], exploration)
    reference = exploratory_values(profiles, exploration)
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("Nicht endliche Werte bei der Zentroidprojektion.")
    distances = cdist(values, reference, metric="euclidean")
    nearest = distances.argmin(axis=1)
    result["map_cell_id"] = cells.iloc[nearest].cell_id.to_numpy()
    result[["component_1", "component_2"]] = cells.iloc[nearest][["component_1", "component_2"]].to_numpy()
    return result


def aggregate_svm_frequencies(records, cells, splits):
    """Alle Testmodelle je Spender zählen; ohne Testauftritt bleibt die Häufigkeit NaN."""
    if records.duplicated(["split_id", "cell_id"]).any():
        raise ValueError("Doppelte OOF-Zellbewertung.")
    joined = records.merge(cells[["cell_id", "sample_id"]], on="cell_id", how="left", validate="many_to_one")
    allowed = splits.loc[splits.outer_partition.eq("test"), ["split_id", "donor_id"]]
    validation = joined.merge(allowed, left_on=["split_id", "sample_id"], right_on=["split_id", "donor_id"],
                              how="left", validate="many_to_one")
    if validation.donor_id.isna().any():
        raise ValueError("Eine Zellbewertung verwendet einen Nicht-Testspender oder eine unbekannte Zelle.")
    counts = joined.groupby("cell_id").agg(n_test_models=("split_id", "size"),
                                           positive_count=("positive", "sum"), negative_count=("negative", "sum"))
    result = cells.merge(counts, on="cell_id", how="left", validate="one_to_one")
    for column in counts.columns:
        result[column] = result[column].fillna(0).astype(int)
    denominators = result.sample_id.map(allowed.groupby("donor_id").size()).fillna(0)
    if not result.n_test_models.eq(denominators).all():
        raise ValueError("OOF-Modelle fehlen; fehlende Ergebnisse sind keine Nullmodelle.")
    for direction in ("positive", "negative"):
        result[f"{direction}_frequency"] = result[f"{direction}_count"] / result.n_test_models.replace(0, np.nan)
    return result


def svm_frequencies(inputs, artifacts):
    """Unveränderte SVM-Auswahl auf vollständigen Testspendern; Ausgabe nur für Karten-Zellen."""
    records = []
    for split_id in artifacts["split_ids"]:
        split = artifacts["splits"].loc[artifacts["splits"].split_id.eq(split_id)]
        parameters = artifacts["svm_models"].loc[artifacts["svm_models"].split_id.eq(split_id)].set_index("marker").loc[inputs["markers"]]
        scaler = restore_scaler(parameters)
        threshold = float(parameters.decision_threshold.iloc[0])
        for donor in sorted(split.loc[split.outer_partition.eq("test"), "donor_id"]):
            margins = scaler.transform(inputs["data"][donor]) @ parameters.weight.to_numpy() + parameters.intercept.iloc[0]
            positive, negative, top = svm_cell_selection(margins, threshold)
            saved = artifacts["svm_predictions"].loc[artifacts["svm_predictions"].split_id.eq(split_id) & artifacts["svm_predictions"].donor_id.eq(donor)].iloc[0]
            if abs(margins[top].mean() - saved.score) > 1e-10 or threshold != saved.decision_threshold or len(top) != saved.top_cell_count:
                raise ValueError("SVM: gepoolte Margins passen nicht zur gespeicherten Vorhersage.")
            cells = inputs["cells"].loc[inputs["cells"].sample_id.eq(donor)]
            indices = cells.event_index.to_numpy()
            records.append(pd.DataFrame(dict(split_id=split_id, cell_id=cells.cell_id,
                                             positive=positive[indices], negative=negative[indices])))
    return aggregate_svm_frequencies(pd.concat(records, ignore_index=True), inputs["cells"], artifacts["splits"])


def representative_cells(inputs, artifacts, centroids, groups):
    """Häufigste wiederkehrende CellCNN-Gruppe explorativ auf allen Karten-Zellen zeigen."""
    candidates = groups.loc[groups.method.eq("CellCNN") & groups.retained].sort_values("group_id")
    columns = list(inputs["cells"].columns) + ["group_id", "split_id", "subset_id", "response", "selected"]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    group = candidates.iloc[0]
    centroid = centroids.loc[centroids.method.eq("CellCNN") &
                             centroids.split_id.eq(group.representative_split_id) &
                             centroids.subset_id.eq(group.representative_subset_id)].iloc[0]
    filters = artifacts["cellcnn_filters"].loc[artifacts["cellcnn_filters"].split_id.eq(centroid.split_id)]
    # Alle Filter gemeinsam auswerten wie bei der Referenz; danach denselben Filter auswählen.
    values = np.stack([inputs["data"][r.sample_id][r.event_index] for r in inputs["cells"].itertuples()])
    index = sorted(filters.filter_id.unique()).index(centroid.subset_id)
    responses = filter_responses(values, filters, inputs["markers"])[:, index]
    return inputs["cells"].assign(group_id=int(group.group_id), split_id=int(centroid.split_id),
                                  subset_id=int(centroid.subset_id), response=responses,
                                  selected=responses > centroid.response_threshold)


def interpret_models(inputs, artifacts):
    """Gemeinsame exportfreie Pipeline; explizite Teilmengen nur für technische Tests."""
    torch.set_num_threads(1)
    cnn = cellcnn_centroids(inputs, artifacts)
    citrus = citrus_centroids(artifacts, inputs["markers"])
    # Einheitliche Spalten auch für vollständig leere Methoden.
    parts = [frame for frame in (cnn, citrus) if not frame.empty]
    centroids = pd.concat(parts, ignore_index=True) if parts else cnn.copy()
    centroids, groups = group_centroids(centroids, inputs["markers"], inputs["exploration"], artifacts["split_ids"])
    centroids = project_centroids(centroids, inputs["cells"], inputs["profiles"], inputs["markers"], inputs["exploration"])
    return dict(centroids=centroids, groups=groups, svm_cells=svm_frequencies(inputs, artifacts),
                representative_cells=representative_cells(inputs, artifacts, centroids, groups))


def run_interpretation(root):
    """Nur vollständige 30-Split-Auswertung unter task5_paper_* exportieren."""
    artifacts = load_artifacts(root)  # Fehlende Splits vor dem teuren Lesen der FCS-Dateien melden.
    inputs = load_inputs(root)
    outputs = interpret_models(inputs, artifacts)
    tables = inputs["tables"]
    # Ergebnisse erst nach vollständig erfolgreicher Auswertung schreiben.
    for name, frame in outputs.items():
        frame.to_csv(tables / f"task5_paper_{name}.csv", index=False)
    source_hashes = dict(artifacts["source_hashes"])
    for name in ("cells.csv", "embeddings.csv", "provenance.json"):
        path = tables / f"task2_{name}"
        source_hashes[path.name] = file_hash(path)
    provenance = dict(
        gate="gated_alive", split_ids=list(SPLIT_IDS), markers=inputs["markers"],
        selected_data_sha256=inputs["selected_data_sha256"], input_sha256=inputs["input_hashes"],
        artifact_sha256=source_hashes, implementation_sha256=file_hash(Path(__file__)),
        package_versions={name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "flowkit")},
        cellcnn_reference="original fit_balanced_scaler draw; 20000 cells per actual inner-training donor; original candidate seed",
        cellcnn_selection="ReLU response > 0.5 * reference maximum; nonzero output contrast",
        centroid_space="arcsinh(x/5); CellCNN model input float32, means accumulated in float64; Citrus saved centroids",
        exploratory_scaling="stored task2 mean/scale; only retrospective grouping and projection, never classifier input",
        clustering=dict(linkage="average", metric="cosine", distance_cutoff=.4,
                        source_commit="0413a9f49fe0831c8fe3280957fb341f9e028d2d",
                        adaptation="filter-grouping rule transferred to subset centroids; not specified for NK centroids in paper"),
        stability="unique split count / 30, including null models; retain >= 6 splits",
        representative="minimum summed cosine distance; ties within 1e-12 by split/subset ID; all map cells exploratory",
        projection="existing tsne_p30; nearest map cell by Euclidean distance in exploratory z-space",
        svm_selection="exact top ceil(1% full test donor); sign(margin - saved donor threshold); ties event index",
        svm_denominator="all outer test appearances of donor in splits 0–29",
        output_sha256={f"task5_paper_{name}.csv": file_hash(tables / f"task5_paper_{name}.csv") for name in outputs},
    )
    (tables / "task5_paper_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return outputs
