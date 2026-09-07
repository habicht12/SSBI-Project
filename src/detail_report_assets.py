"""Deutsche Berichtsabbildungen und unabhängiger Tabellenabgleich, ohne Modellfit.

Aufruf im Projektstamm: python -m src.detail_report_assets
Alle Ausgaben liegen unter report/detail_assets; Ergebnisse und FCS bleiben unverändert.
"""
from pathlib import Path
import hashlib
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns

from src.report_assets import align_cells

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["CellCNN", "Citrus", "SVM"]
COLOURS = ["#0072B2", "#D58000", "#009E73"]
SHORT = {"CellCNN": "CellCNN", "Citrus": "Citrus", "Lineare Single-Cell-SVM": "SVM"}
VARIANTS = ["pca", "pca_white", "tsne_p5", "tsne_p30", "tsne_p50",
            "umap_n15_d0.1", "umap_n50_d0.1", "umap_n15_d0.5"]
TITLES = ["PCA", "PCA mit Whitening", "t-SNE: p = 5", "t-SNE: p = 30", "t-SNE: p = 50",
          "UMAP: n = 15, d = 0,1", "UMAP: n = 50, d = 0,1", "UMAP: n = 15, d = 0,5"]
PROFILE_MARKERS = ["CD3", "CD4", "CD8", "CD19", "CD33", "CD11b", "CD56", "CD16",
                   "CD94", "NKG2A", "NKG2C", "CD57"]


def independent_metrics(labels, scores, predictions):
    """Rangpaare und gruppierte Score-Schwellen; unabhängig von sklearn.metrics."""
    y, s, pred = np.asarray(labels), np.asarray(scores), np.asarray(predictions)
    if (y.ndim != 1 or y.shape != s.shape or y.shape != pred.shape
            or set(y) != {0, 1} or not np.isfinite(s).all() or not set(pred) <= {0, 1}):
        raise ValueError("Endliche binäre Daten beider Klassen mit gleicher Länge erforderlich.")
    differences = s[y == 1, None] - s[y == 0][None, :]
    roc = float(((differences > 0) + .5 * (differences == 0)).mean())
    precision, recall = [1.], [0.]
    for threshold in np.unique(s)[::-1]:
        positive = s >= threshold
        precision.append(float(y[positive].mean()))
        recall.append(float(y[positive].sum() / y.sum()))
    precision, recall = np.array(precision), np.array(recall)
    return dict(roc_auc=roc, average_precision=float(np.sum(np.diff(recall) * precision[1:])),
                pr_auc=float(np.trapezoid(precision, recall)),
                balanced_accuracy=float(.5 * (pred[y == 1].mean() + (1 - pred[y == 0]).mean())))


def independent_selection(scores, splits):
    """Häufigkeitsnenner, Zellidentitäten und selektierte Kartenpopulationen abgleichen."""
    required = splits.loc[splits.outer_partition.eq("test")].groupby("donor_id").size()
    if scores.duplicated(["method", "cell_id"]).any():
        raise ValueError("Doppelte Zellbewertungen.")
    if not scores.n_test_models.eq(scores.sample_id.map(required)).all():
        raise ValueError("Falscher Testmodellnenner.")
    rows = []
    for method, frame in scores.groupby("method", sort=False):
        if len(frame) != 10000 or not frame.groupby("sample_id").size().eq(500).all():
            raise ValueError("Unvollständige Kartenstichprobe.")
        for direction in ["positive", "negative", "ambiguous"]:
            count, freq = frame[f"{direction}_count"], frame[f"{direction}_frequency"]
            if not count.between(0, frame.n_test_models).all():
                raise ValueError("Ungültige Auswahlzählung.")
            np.testing.assert_allclose(freq, count / frame.n_test_models, rtol=0, atol=1e-15)
            if direction == "ambiguous":
                continue
            by_donor = frame.assign(chosen=freq > 0).groupby("sample_id").agg(
                selected_cells=("chosen", "sum"), mean_frequency=(f"{direction}_frequency", "mean"))
            rows.append(dict(method=method, direction=direction,
                             selected_cells=int(by_donor.selected_cells.sum()),
                             supported_donors=int((by_donor.selected_cells > 0).sum()),
                             mean_selection_fraction=by_donor.mean_frequency.mean(),
                             minimum_donor_cells=int(by_donor.selected_cells.min()),
                             maximum_donor_cells=int(by_donor.selected_cells.max())))
    if set(scores.method) != set(METHODS):
        raise ValueError("Methoden fehlen.")
    return pd.DataFrame(rows)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def number(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


def tex(value):
    return str(value).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("−", "$-$")


def table(out, name, headers, rows, alignment=None):
    align = alignment or "l" + "r" * (len(headers) - 1)
    text = [r"\begin{tabular}{" + align + "}", r"\toprule",
            " & ".join(map(tex, headers)) + r" \\", r"\midrule"]
    text += [" & ".join(map(tex, row)) + r" \\" for row in rows]
    text += [r"\bottomrule", r"\end{tabular}"]
    (out / f"{name}.tex").write_text("\n".join(text) + "\n")


def save(fig, out, name):
    # Die PDF wird auf etwa 16 cm gesetzt: Beschriftungen müssen dort lesbar bleiben.
    from matplotlib.text import Text
    for artist in fig.findobj(Text):
        artist.set_fontsize(max(13.5, artist.get_fontsize()))
    if name not in {"projektionen", "nk_koexpression", "selektion_positive", "selektion_negative"}:
        fig.tight_layout()
    fig.savefig(out / f"{name}.pdf", dpi=240, bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def read_measurements(root, tables):
    """Erneute read-only QC; identische QC-, NK- und Kartenstichproben, kein Fit."""
    import flowkit as fk
    from untersuchung.inspect_fcs import read_fcs_metadata

    data = root / "NK_cell_dataset/NK_cell_dataset"
    markers = pd.read_csv(data / "NK_markers.csv", header=None).iloc[0].tolist()
    labels = pd.read_csv(data / "NK_fcs_samples_with_labels.csv")
    labels["donor_id"] = labels.fcs_filename.str.replace("_NK.fcs", "", regex=False)
    labels = labels.sort_values("donor_id").reset_index(drop=True)
    cells = pd.read_csv(tables / "task2_cells.csv")
    rng = np.random.default_rng(12345)
    raw_map, qc, nk, overview = [], [], [], []
    paths = []
    for index, row in labels.iterrows():
        counts = {}
        for gate, suffix in [("gated_alive", "alive"), ("gated_NK", "NK")]:
            path = data / "NK_cell_dataset" / gate / f"{row.donor_id}_{suffix}.fcs"
            paths.append(path)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"FCS file .* reported incorrect data offset.*")
                sample = fk.Sample(str(path), ignore_offset_error=True)
            names = sample.pns_labels
            if len(set(names)) != len(names) or not set(markers) <= set(names):
                raise ValueError(f"Fehlende oder doppelte Marker: {path.name}")
            raw = sample.get_events(source="raw")[:, [names.index(m) for m in markers]]
            if not np.isfinite(raw).all():
                raise ValueError(f"Nicht endliche Messwerte: {path.name}")
            counts[suffix] = len(raw)
            if read_fcs_metadata(path)["events"] != len(raw):
                raise ValueError("Abweichende FCS-Eventzahl.")
            if gate == "gated_alive":
                selected = cells.loc[cells.sample_id.eq(row.donor_id)]
                expected = np.sort(np.random.default_rng(42 + index).choice(len(raw), 500, replace=False))
                np.testing.assert_array_equal(selected.event_index, expected)
                raw_map.append(raw[expected].astype(np.float64))
                positions = rng.choice(len(raw), 2000, replace=False)
                frame = pd.DataFrame(np.arcsinh(raw.astype(np.float32)[positions] / 5), columns=markers)
                frame["donor_id"] = row.donor_id
                qc.append(frame)
                date, instrument = sample.metadata.get("date", ""), sample.metadata.get("cyt", "")
            else:
                frame = pd.DataFrame(raw, columns=markers).sample(n=1000, random_state=42 + index)
                frame = np.arcsinh(frame / 5)
                frame["group"] = "CMV+" if row.label else "CMV−"
                nk.append(frame)
        overview.append(dict(donor_id=row.donor_id, label=row.label, alive=counts["alive"],
                             nk=counts["NK"], date=date, instrument=instrument))
    raw_map = np.concatenate(raw_map)
    provenance = json.loads((tables / "task2_provenance.json").read_text())
    h = hashlib.sha256(json.dumps({"cell_ids": cells.cell_id.tolist(), "markers": markers}, separators=(",", ":")).encode())
    h.update(raw_map.astype("<f8").tobytes())
    if h.hexdigest() != provenance["selected_data_sha256"]:
        raise ValueError("Karten-Rohwerte stimmen nicht zur Provenienz.")
    return cells, markers, np.arcsinh(raw_map / 5), pd.concat(qc), pd.concat(nk), pd.DataFrame(overview), paths


def audit_tables(root, tables, cells, markers, profiles, raw_paths):
    read = lambda n: pd.read_csv(tables / f"{n}.csv", float_precision="round_trip")
    splits = read("task4_donor_splits")
    if len(splits) != 2000 or splits.duplicated(["split_id", "donor_id"]).any() or set(splits.split_id) != set(range(100)):
        raise ValueError("Ungültige zentrale Splits.")
    original = pd.read_csv(root / "NK_cell_dataset/NK_cell_dataset/NK_fcs_samples_with_labels.csv")
    labels = dict(zip(original.fcs_filename.str.replace("_NK.fcs", "", regex=False), original.label))
    if not splits.label.eq(splits.donor_id.map(labels)).all():
        raise ValueError("Labels widersprechen Originaldatei.")
    for _, group in splits.groupby("split_id"):
        observed = group.groupby(["outer_partition", "label"]).size().to_dict()
        if observed != {("test", 0): 4, ("test", 1): 2, ("train", 0): 7, ("train", 1): 7}:
            raise ValueError("Falsche Klassenaufteilung.")
        if not group.loc[group.outer_partition.eq("test"), "inner_fold"].eq(-1).all():
            raise ValueError("Testspender hat inneren Fold.")
        inner = group.loc[group.outer_partition.eq("train")].groupby(["inner_fold", "label"]).size()
        if len(inner) != 6 or not inner.between(2, 3).all():
            raise ValueError("Ungültige innere Folds.")
    saved = read("task4_split_metrics")
    saved["method"] = saved.method_label.map(SHORT)
    records = []
    for method in METHODS:
        pred = read(f"task4_{method.lower()}_predictions_gated_alive_full")
        expected = splits.loc[splits.outer_partition.eq("test")]
        joined = pred.merge(expected, on=["split_id", "donor_id"], validate="one_to_one", suffixes=("", "_expected"))
        if len(joined) != len(pred) or len(pred) != 600 or not joined.y_true.eq(joined.label).all() or not joined.split_seed.eq(joined.split_seed_expected).all():
            raise ValueError("Vorhersagezuordnung stimmt nicht.")
        if not pred.gate.eq("gated_alive").all() or not pred.run_mode.eq("full").all():
            raise ValueError("Falsches Gate/Modus.")
        np.testing.assert_array_equal(pred.y_pred, (pred.score >= pred.decision_threshold).astype(int))
        for split, frame in pred.groupby("split_id"):
            metrics = independent_metrics(frame.y_true, frame.score, frame.y_pred)
            existing = saved.loc[saved.method.eq(method) & saved.split_id.eq(split)].iloc[0]
            np.testing.assert_allclose(list(metrics.values()), existing[list(metrics)].to_numpy(float), atol=1e-12, rtol=0)
            records.append(dict(method=method, split_id=split, **metrics))
    metrics = pd.DataFrame(records)
    metric_names = ["roc_auc", "average_precision", "pr_auc", "balanced_accuracy"]
    summary = read("task4_metric_summary")
    for row in summary.itertuples():
        values = metrics.loc[metrics.method.eq(SHORT[row.Methode]), row.Metrik]
        np.testing.assert_allclose([values.median(), values.quantile(.25), values.quantile(.75), values.mean(), values.std()],
                                   [row.Median, row.Q1, row.Q3, row.Mittelwert, row.Standardabweichung], atol=1e-12, rtol=0)
    cnn = read("task4_cellcnn_selection_gated_alive_full")
    filters = read("task4_cellcnn_filters_gated_alive_full")
    for split, frame in cnn.groupby("split_id"):
        if len(frame) != 9 or frame.duplicated(["inner_fold", "filter_count"]).any() or frame.selected.sum() != 1:
            raise ValueError("Unvollständige CellCNN-Auswahl.")
        winner = frame.sort_values(["validation_accuracy", "validation_roc_auc", "best_validation_loss", "filter_count", "inner_fold"], ascending=[False, False, True, True, True]).iloc[0]
        if not winner.selected:
            raise ValueError("CellCNN-Auswahlregel verletzt.")
        f = filters.loc[filters.split_id.eq(split)]
        if len(f) != 37 * winner.filter_count or not f.inner_fold.eq(winner.inner_fold).all():
            raise ValueError("CellCNN-Parameter nicht vollständig.")
    svm_selection = read("task4_svm_selection_gated_alive_full")
    svm_models = read("task4_svm_models_gated_alive_full")
    for split, frame in svm_selection.groupby("split_id"):
        expected_candidates = {(c, f) for c in [.01, .1, 1.] for f in range(3)}
        if len(frame) != 9 or set(map(tuple, frame[["C", "inner_fold"]].values)) != expected_candidates:
            raise ValueError("Unvollständige SVM-Kandidaten.")
        means = frame.groupby("C").validation_roc_auc.mean()
        best = min(means.index[means == means.max()])
        if not frame.selected.eq(frame.C.eq(best)).all():
            raise ValueError("SVM-Auswahlregel verletzt.")
        parameters = svm_models.loc[svm_models.split_id.eq(split)]
        if len(parameters) != 37 or set(parameters.marker) != set(markers) or not parameters.best_C.eq(best).all():
            raise ValueError("SVM-Modellparameter stimmen nicht zur Auswahl.")
    citrus_selection = read("task4_citrus_selection_gated_alive_full")
    clusters = read("task4_citrus_clusters_gated_alive_full")
    if not clusters.coefficient.abs().gt(1e-10).all() or clusters.duplicated(["split_id", "cluster_id", "marker"]).any():
        raise ValueError("Ungültige Citrus-Clusterparameter.")
    cluster_counts = clusters[["split_id", "cluster_id"]].drop_duplicates().groupby("split_id").size()
    for row in citrus_selection.itertuples():
        if row.selected_cluster_count != cluster_counts.get(row.split_id, 0):
            raise ValueError("Citrus-Clusterzählung stimmt nicht.")
    for method, notebook, indices in [("svm", "04b_svm", [4, 6, 8, 13]), ("cellcnn", "04c_cellcnn", [4, 6, 8, 10])]:
        config = json.loads((tables / f"task4_{method}_predictions_gated_alive_full.config.json").read_text())
        content = json.loads((root / f"notebooks/{notebook}.ipynb").read_text())
        source = "\n\n".join("".join(content["cells"][i]["source"]) for i in indices)
        if hashlib.sha256(source.encode()).hexdigest() != config["implementation_sha256"]:
            raise ValueError("Notebookcode passt nicht zum gespeicherten Benchmark.")
    selection = read("task5_cell_scores")
    for method in METHODS:
        frame = align_cells(selection.loc[selection.method.eq(method)], cells)
        projection = align_cells(read("task2_embeddings").query("variant == 'tsne_p30'"), cells)
        np.testing.assert_allclose(frame[["component_1", "component_2"]], projection[["component_1", "component_2"]], atol=1e-12)
    selected = independent_selection(selection, splits)
    columns = [c for c in selected if c not in ["method", "direction"]]
    a = selected.set_index(["method", "direction"]).sort_index()
    b = read("task5_selection_summary").set_index(["method", "direction"]).sort_index()
    np.testing.assert_allclose(a[columns], b[columns], atol=1e-12, rtol=0)
    # Unabhängige gewichtete Quantilsberechnung direkt aus Kartenwerten.
    donor_rows = []
    values = pd.DataFrame(profiles, index=cells.cell_id, columns=markers)
    for method in METHODS:
        frame = selection.loc[selection.method.eq(method)]
        for direction in ["positive", "negative"]:
            for donor, group in frame.groupby("sample_id"):
                weights = group[f"{direction}_frequency"].to_numpy()
                for marker in markers:
                    v = values.loc[group.cell_id, marker].to_numpy()
                    order = np.argsort(v[weights > 0], kind="stable")
                    v, w = v[weights > 0][order], weights[weights > 0][order]
                    median = v[np.flatnonzero(np.cumsum(w) >= w.sum() / 2)[0]] if len(w) else np.nan
                    donor_rows.append([method, direction, donor, marker, median])
    for donor, group in cells.groupby("sample_id"):
        for marker in markers:
            donor_rows.append(["Kartenreferenz", "all", donor, marker, values.loc[group.cell_id, marker].median()])
    donor_profiles = pd.DataFrame(donor_rows, columns=["method", "direction", "donor_id", "marker", "value"])
    keys = ["method", "direction", "donor_id", "marker"]
    np.testing.assert_allclose(donor_profiles.set_index(keys).sort_index().value,
                               read("task5_donor_marker_profiles").set_index(keys).sort_index().value, atol=1e-12, rtol=0)
    grouped = donor_profiles.groupby(["method", "direction", "marker"]).value.median()
    np.testing.assert_allclose(grouped, read("task5_marker_profiles").set_index(["method", "direction", "marker"]).sort_index().value, atol=1e-12, rtol=0)
    provenance = json.loads((tables / "task5_provenance.json").read_text())
    input_paths = {p.name: p for p in raw_paths}
    data = root / "NK_cell_dataset/NK_cell_dataset"
    input_paths.update({p.name: p for p in [data / "NK_fcs_samples_with_labels.csv", data / "NK_markers.csv", tables / "task4_donor_splits.csv"]})
    hash_count = 0
    for key, directory in [("input_sha256", None), ("artifact_sha256", tables), ("implementation_sha256", root / "src"), ("output_sha256", tables)]:
        for name, digest in provenance[key].items():
            path = directory / name if directory is not None else input_paths[name]
            if sha(path) != digest:
                raise ValueError(f"Veralteter Aufgabe-5-Nachweis: {path.name}")
            hash_count += 1
    checks = read("task5_prediction_checks")
    if checks.groupby("method").size().to_dict() != {m: 600 for m in METHODS} or not checks.decision_matches.all():
        raise ValueError("Unvollständige historische Rechenprüfungen.")
    audit = dict(metrics_recomputed=1200, donor_profile_values_recomputed=len(donor_profiles),
                 provenance_hashes_checked=hash_count, historical_prediction_checks=len(checks),
                 classifier_fits=0, projection_fits=0, cluster_fits=0,
                 note="Tabellen/QC/Hashes jetzt geprüft; Modellinferenz-Prüfungen aus hashverifiziertem Aufgabe-5-Artefakt.")
    return metrics, selected, audit


def export(root=ROOT):
    tables = root / "results/tables"
    out = root / "report/detail_assets"
    out.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(tables / f"{name}.csv", float_precision="round_trip")
    sns.set_theme(style="whitegrid", font_scale=.9)
    plt.rcParams.update({"pdf.fonttype": 42, "axes.titlesize": 11, "axes.labelsize": 10})
    cells, markers, values, qc, nk, donors, raw_paths = read_measurements(root, tables)
    print("Read-only QC und Kartenidentität geprüft.", flush=True)
    metrics, selection, audit = audit_tables(root, tables, cells, markers, values, raw_paths)
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    (out / "pruefung.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    # Datenumfang und ursprüngliche NK-Exploration.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [2.3, 1]})
    donors.set_index("donor_id")[["alive", "nk"]].rename(columns={"alive": "Alive", "nk": "NK"}).plot.bar(ax=axes[0], logy=True, color=["#4477AA", "#EE9944"])
    axes[0].set(xlabel="Spender", ylabel="Zellzahl (logarithmisch)")
    donors["group"] = donors.label.map({0: "CMV−", 1: "CMV+"})
    donors["nk_fraction"] = 100 * donors.nk / donors.alive
    sns.boxplot(data=donors, x="group", y="nk_fraction", ax=axes[1], color="#B8D7EE")
    sns.stripplot(data=donors, x="group", y="nk_fraction", jitter=False, color="black", ax=axes[1])
    axes[1].set(xlabel="CMV-Gruppe", ylabel="NK / Alive [%]")
    save(fig, out, "zellzahlen")
    table(out, "spender", ["Spender", "Label", "Alive", "NK", "NK/Alive (%)", "Testmodelle"],
          [[r.donor_id, r.label, r.alive, r.nk, number(r.nk_fraction, 2),
            int(read("task4_donor_splits").query("outer_partition == 'test'").donor_id.eq(r.donor_id).sum())] for r in donors.itertuples()])
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.6))
    for ax, marker in zip(axes.flat, ["CD3", "CD56", "CD57", "NKG2C", "CD16", "CD94"]):
        sns.histplot(nk, x=marker, hue="group", bins=45, stat="density", common_norm=False, element="step", fill=False, ax=ax)
        ax.set(xlabel=f"arcsinh({marker} / 5)", ylabel="Dichte")
        if ax.get_legend(): ax.get_legend().set_title("Gruppe")
    fig.tight_layout(); save(fig, out, "nk_histogramme")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True, sharey=True)
    collections = []
    for ax, group in zip(axes, ["CMV−", "CMV+"]):
        f = nk.loc[nk.group.eq(group)]
        collections.append(ax.hexbin(f.NKG2C, f.CD57, gridsize=45, mincnt=1, cmap="viridis"))
        ax.set(title=f"{group}: {len(f):,} Zellen".replace(",", "."), xlabel="arcsinh(NKG2C / 5)", ylabel="arcsinh(CD57 / 5)")
    norm = LogNorm(1, max(c.get_array().max() for c in collections))
    for c in collections: c.set_norm(norm)
    fig.colorbar(collections[0], ax=axes, label="Zellen pro Hexagon (log. Skala)", pad=.02)
    save(fig, out, "nk_koexpression")
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), gridspec_kw={"width_ratios": [1, 1.4]})
    sns.boxplot(qc[markers], orient="h", showfliers=False, color="#6699BB", ax=axes[0], linewidth=.5)
    axes[0].set(xlabel="arcsinh(Messwert / 5)", ylabel="Marker")
    medians = qc.groupby("donor_id")[markers].median()
    center = medians.median(); mad = medians.sub(center).abs().median()
    sns.heatmap(medians.sub(center).div(1.4826 * mad).T, vmin=-3, vmax=3, center=0, cmap="vlag", ax=axes[1], cbar_kws={"label": "Robuster z-Wert des Spender-Medians"})
    axes[1].set(xlabel="Spender", ylabel="")
    axes[1].tick_params(axis="y", labelsize=8); axes[0].tick_params(axis="y", labelsize=8)
    fig.tight_layout(); save(fig, out, "qc_marker")
    # Acht unveränderte Projektionen und vollständige Paarmatrix.
    embeddings = read("task2_embeddings")
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.8))
    cd3 = values[:, markers.index("CD3")]
    order = np.random.default_rng(42).permutation(len(cells))
    for ax, variant, title in zip(axes.flat, VARIANTS, TITLES):
        f = align_cells(embeddings.loc[embeddings.variant.eq(variant)], cells)
        scatter = ax.scatter(f.component_1.iloc[order], f.component_2.iloc[order], c=cd3[order], s=1.2, cmap="viridis", vmin=cd3.min(), vmax=cd3.max(), rasterized=True)
        ax.set(title=title.replace(", d =", "\nd =") if variant.startswith("umap") else title,
               xticks=[], yticks=[], aspect="equal")
    fig.subplots_adjust(bottom=.13, hspace=.40, wspace=.16)
    cax = fig.add_axes([.28, .035, .44, .02]); fig.colorbar(scatter, cax=cax, orientation="horizontal", label="CD3: arcsinh(Messwert / 5)")
    save(fig, out, "projektionen")
    pairs = read("task2_pairwise_jaccard")
    matrix = pd.DataFrame(np.eye(8), index=VARIANTS, columns=VARIANTS)
    if len(pairs) != 28: raise ValueError("Kartenpaare fehlen.")
    for r in pairs.itertuples(): matrix.loc[r.first, r.second] = matrix.loc[r.second, r.first] = r.neighbor_jaccard
    fig, ax = plt.subplots(figsize=(7, 5.4))
    sns.heatmap(matrix, annot=True, fmt=".2f", square=True, cmap="viridis", vmin=0, vmax=1,
                xticklabels=["P", "Pw", "T5", "T30", "T50", "U15", "U50", "Ud"],
                yticklabels=["P", "Pw", "T5", "T30", "T50", "U15", "U50", "Ud"], ax=ax,
                cbar_kws={"label": "Spendergemittelter Jaccard-Index"})
    save(fig, out, "nachbarschaften")
    ref = read("task2_reference_jaccard").set_index("variant")
    table(out, "projektion_werte", ["Einstellung", "Jaccard zu 37D"], [[t, number(ref.loc[v, "reference_jaccard"], 5)] for v, t in zip(VARIANTS, TITLES)])
    # Clustering: identische gespeicherte UMAP, keine neue Geometrie.
    clustering = read("task3_cluster_comparison")
    assignments = align_cells(read("task3_assignments"), cells)
    umap = align_cells(read("task3_umap"), cells)
    chosen = clustering.loc[clustering.selected]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, r in zip(axes, chosen.itertuples()):
        ax.scatter(umap.component_1.iloc[order], umap.component_2.iloc[order], c=assignments[r.variant].iloc[order], cmap="tab20", vmin=0, vmax=19, s=1, rasterized=True)
        ax.set(title=f"{r.method}: {r.n_clusters} Cluster", xticks=[], yticks=[], aspect="equal")
    save(fig, out, "clusterkarten")
    table(out, "clustervergleich", ["Methode", "Parameter", "Cluster", "Min. Zellen", "Silhouette", "Auswahl"],
          [[r.method, r.variant.split("_")[-1], r.n_clusters, r.min_cluster_size, number(r.silhouette, 4), "ja" if r.selected else ""] for r in clustering.itertuples()], "llrrrl")
    cp = read("task3_marker_profiles")
    prov = json.loads((tables / "task3_provenance.json").read_text())
    cp[markers] = (cp[markers] - np.array(prov["scaler_mean"])) / np.array(prov["scaler_scale"])
    limit = max(1., abs(cp[PROFILE_MARKERS].to_numpy()).max())
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 5))
    for ax, r in zip(axes, chosen.itertuples()):
        sns.heatmap(cp.loc[cp.variant.eq(r.variant)].set_index("cluster")[PROFILE_MARKERS], cmap="vlag", vmin=-limit, vmax=limit, center=0, ax=ax, cbar=ax is axes[-1], cbar_kws={"label": "Explorativer z-Wert"})
        ax.set(title=r.method, xlabel="", ylabel="Cluster-ID"); ax.tick_params(axis="x", rotation=90)
    fig.tight_layout(); save(fig, out, "clusterprofile")
    # Kandidaten-, Leistungs- und Interpretationszusammenfassungen.
    cnn = read("task4_cellcnn_selection_gated_alive_full")
    sel = cnn.loc[cnn.selected]
    table(out, "cellcnn_auswahl", ["Filterzahl", "Fold 0", "Fold 1", "Fold 2", "Gesamt"],
          [[k, *[int(((sel.filter_count == k) & (sel.inner_fold == f)).sum()) for f in range(3)], int((sel.filter_count == k).sum())] for k in [3, 4, 5]])
    table(out, "cellcnn_split0", ["Fold", "Filter", "Accuracy", "ROC-AUC", "Bester Verlust", "Epochen", "Gewählt"],
          [[r.inner_fold, r.filter_count, number(r.validation_accuracy), number(r.validation_roc_auc), number(r.best_validation_loss, 4), r.epochs_run, "ja" if r.selected else ""] for r in cnn.query("split_id == 0").itertuples()], "rrrrrrl")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].hist([cnn.epochs_run, sel.epochs_run], bins=[0, 10, 20, 40, 60, 80, 101], label=["Alle 900 Kandidaten", "100 ausgewählte Netze"], color=["#99BBDD", "#0072B2"])
    axes[0].set(xlabel="Durchlaufene Epochen", ylabel="Anzahl"); axes[0].legend(fontsize=8)
    cit = read("task4_citrus_selection_gated_alive_full")
    axes[1].hist(cit.selected_cluster_count, bins=np.arange(cit.selected_cluster_count.max() + 2) - .5, color=COLOURS[1])
    axes[1].set(xlabel="Wirksame Citrus-Cluster", ylabel="Anzahl äußerer Splits")
    fig.tight_layout(); save(fig, out, "modellauswahl")
    metric_labels = {"roc_auc": "ROC-AUC", "average_precision": "AP", "pr_auc": "PR-AUC (Trapez)", "balanced_accuracy": "Balanced Accuracy"}
    table(out, "metriken", ["Methode", "Metrik", "Median", "Q1", "Q3", "Mittel", "SD"],
          [[m, label, *[number(v) for v in [s.median(), s.quantile(.25), s.quantile(.75), s.mean(), s.std()]]]
           for m in METHODS for name, label in metric_labels.items() for s in [metrics.loc[metrics.method.eq(m), name]]], "llrrrrr")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    sns.boxplot(metrics, x="method", y="roc_auc", order=METHODS, hue="method", palette=dict(zip(METHODS, COLOURS)), legend=False, ax=ax)
    sns.stripplot(metrics, x="method", y="roc_auc", order=METHODS, jitter=False, color="black", alpha=.25, size=3, ax=ax)
    ax.axhline(.5, color="gray", ls="--"); ax.set(xlabel="Methode", ylabel="ROC-AUC", ylim=(-.03, 1.03))
    save(fig, out, "roc_auc")
    wide = metrics.pivot(index="split_id", columns="method", values="roc_auc")
    differences = pd.DataFrame({"CellCNN − Citrus": wide.CellCNN - wide.Citrus, "CellCNN − SVM": wide.CellCNN - wide.SVM})
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.boxplot(differences, ax=ax, color="#AACCDD"); sns.stripplot(differences, jitter=False, color="black", size=3, alpha=.3, ax=ax)
    ax.axhline(0, color="gray", ls="--"); ax.set(ylabel="Gepaarte Differenz der ROC-AUC")
    save(fig, out, "auc_differenzen")
    table(out, "differenzen", ["Vergleich", "Median", "Q1", "Q3", "> 0", "= 0", "< 0"],
          [[c, number(differences[c].median()), number(differences[c].quantile(.25)), number(differences[c].quantile(.75)), int((differences[c] > 0).sum()), int((differences[c] == 0).sum()), int((differences[c] < 0).sum())] for c in differences])
    table(out, "selektion", ["Methode", "Richtung", "Zellen", "Spender", "Mittl. Anteil", "Min.–Max."],
          [[r.method, "positiv" if r.direction == "positive" else "negativ", r.selected_cells, r.supported_donors, number(100*r.mean_selection_fraction, 3) + " %", f"{r.minimum_donor_cells}–{r.maximum_donor_cells}"] for r in selection.itertuples()], "llrrrl")
    scores = read("task5_cell_scores")
    for direction in ["positive", "negative"]:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
        for ax, method in zip(axes, METHODS):
            f = align_cells(scores.loc[scores.method.eq(method)], cells)
            idx = np.argsort(f[f"{direction}_frequency"], kind="stable")
            ax.scatter(f.component_1, f.component_2, s=1, color="#CCCCCC", rasterized=True)
            idx = idx[f[f"{direction}_frequency"].to_numpy()[idx] > 0]
            scatter = ax.scatter(f.component_1.iloc[idx], f.component_2.iloc[idx], c=f[f"{direction}_frequency"].iloc[idx], s=2, vmin=0, vmax=1, cmap="viridis", rasterized=True)
            ax.set(title=method, xticks=[], yticks=[], aspect="equal")
        fig.colorbar(scatter, ax=axes, label="Auswahlhäufigkeit", fraction=.025, pad=.02)
        save(fig, out, f"selektion_{direction}")
    profile_table = read("task5_marker_profiles")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4))
    for ax, direction in zip(axes, ["positive", "negative"]):
        f = profile_table.loc[profile_table.direction.isin([direction, "all"])]
        matrix = f.pivot(index="method", columns="marker", values="value").loc[METHODS + ["Kartenreferenz"], PROFILE_MARKERS]
        sns.heatmap(matrix, cmap="viridis", vmin=-.25, vmax=6, annot=True, fmt=".1f", ax=ax, cbar_kws={"label": "arcsinh(Messwert / 5)"})
        ax.set(title="Positive Auswahl" if direction == "positive" else "Negative Auswahl", xlabel="", ylabel="")
        ax.tick_params(axis="x", labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
    fig.tight_layout(); save(fig, out, "selektionsprofile")
    table(out, "markerwerte", ["Population", "CD3", "CD8", "CD56", "NKG2C", "CD57"],
          [[m, *[number(float(profile_table.query("method == @m and direction == @d").set_index("marker").loc[k, "value"])) for k in ["CD3", "CD8", "CD56", "NKG2C", "CD57"]]] for m, d in [(m, "positive") for m in METHODS] + [("Kartenreferenz", "all")]])
    # Verteilung von Testabdeckung und Selektionshäufigkeit statt Gleichsetzung der Regeln.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    coverage = read("task4_donor_splits").query("outer_partition == 'test'").groupby("donor_id").size()
    coverage.plot.bar(ax=axes[0], color="#6699AA"); axes[0].set(xlabel="Spender", ylabel="Äußere Testmodelle")
    for method, colour in zip(METHODS, COLOURS):
        f = np.sort(scores.loc[scores.method.eq(method) & scores.positive_frequency.gt(0), "positive_frequency"])
        axes[1].step(f, np.arange(1, len(f)+1)/len(f), where="post", label=method, color=colour)
    axes[1].set(xlabel="Positive Auswahlhäufigkeit (> 0)", ylabel="Kumulierter Zellanteil", xlim=(0, 1)); axes[1].legend()
    fig.tight_layout(); save(fig, out, "abdeckung")
    checks = read("task5_prediction_checks")
    table(out, "rekonstruktion", ["Methode", "Fälle", "Max. Scorefehler", "Max. Rechenfehler"],
          [[m, 600, "nicht geprüft" if m == "Citrus" else f"{checks.loc[checks.method.eq(m), 'score_error'].max():.2e}", f"{checks.loc[checks.method.eq(m), 'arithmetic_error'].max():.2e}"] for m in METHODS], "lrll")
    # Kurze automatisch erzeugte Kennzahlen für prüfbare Fließtextaussagen.
    facts = dict(AliveCells=int(donors.alive.sum()), NkCells=int(donors.nk.sum()),
                 HashChecks=audit["provenance_hashes_checked"], ProfileChecks=audit["donor_profile_values_recomputed"],
                 CellcnnFilters=int(sel.filter_count.sum()), CnnEpochMedian=number(sel.epochs_run.median(), 1),
                 CnnMaxEpochs=int(sel.epochs_run.eq(100).sum()), NullModels=int(cit.selected_cluster_count.eq(0).sum()))
    (out / "zahlen.tex").write_text("\n".join("\\newcommand{\\" + name + "}{" + str(v) + "}" for name, v in facts.items()) + "\n")
    sources = {p.name: sha(p) for p in sorted(tables.glob("task*.csv")) if "smoke" not in p.name}
    (out / "quellen.json").write_text(json.dumps(dict(inputs=sources, exporter_sha256=sha(Path(__file__))), indent=2) + "\n")
    print(f"Deutsche Berichtsassets: {out}", flush=True)


if __name__ == "__main__":
    export()
