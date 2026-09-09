"""Export task 4/5 slides; read original FCS only for the saved SVM's cell profiles.

Run from the repository root: python -m src.presentation_assets
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from src.report_assets import align_cells, metric_table, validate_interpretation
from src.task4_artifacts import validate_parameter_table, validate_prediction_splits
from src.task5_interpretation import (
    csv_read, load_inputs, restore_scaler, svm_cell_selection,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "praesentation/aufgaben_04_05"
METHODS = ("CellCNN", "Citrus", "Lineare Single-Cell-SVM")
LABELS = ("CellCNN", "Citrus", "Linear SVM")
COLORS = ("#087F8C", "#DD8D29", "#4956A3")
MARKERS = ("CD3", "CD19", "CD56", "CD16", "CD94", "NKG2A", "NKG2C", "CD57")


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_results(tables):
    """Validate source hashes, shared cell identities and plotted quantities."""
    provenance = validate_interpretation(tables)
    exploration = json.loads((tables / "task2_provenance.json").read_text())
    if (exploration["gate"] != "gated_alive"
            or exploration["selected_data_sha256"] != provenance["selected_data_sha256"]):
        raise ValueError("Interpretation and exploration use different data.")
    cells = pd.read_csv(tables / "task2_cells.csv")
    if len(cells) != 10000 or not cells.groupby("sample_id").size().eq(500).all():
        raise ValueError("Expected 500 map cells per donor.")
    embeddings = pd.read_csv(tables / "task2_embeddings.csv")
    reference = align_cells(embeddings.loc[embeddings.variant.eq("tsne_p30")], cells)
    frames = {name: pd.read_csv(tables / f"task5_paper_{name}.csv")
              for name in ("centroids", "groups", "svm_cells", "representative_cells")}
    for name in ("svm_cells", "representative_cells"):
        frames[name] = align_cells(frames[name], cells)
        np.testing.assert_allclose(frames[name][["component_1", "component_2"]],
                                   reference[["component_1", "component_2"]], rtol=0, atol=1e-12)
    centroids, groups = frames["centroids"], frames["groups"]
    mapped = reference.set_index("cell_id").loc[centroids.map_cell_id]
    np.testing.assert_allclose(centroids[["component_1", "component_2"]],
                               mapped[["component_1", "component_2"]], rtol=0, atol=1e-12)
    for group in groups.itertuples():
        members = centroids.loc[centroids.method.eq(group.method) & centroids.group_id.eq(group.group_id)]
        if (members.split_id.nunique() != group.occurrences or group.n_splits != 30
                or group.retained != (group.occurrences >= 6)):
            raise ValueError("Group recurrence disagrees with the saved centroids.")
    splits = pd.read_csv(tables / "task4_donor_splits.csv")
    appearances = splits.loc[splits.split_id.lt(30) & splits.outer_partition.eq("test")].groupby("donor_id").size()
    svm = frames["svm_cells"]
    if not svm.n_test_models.eq(svm.sample_id.map(appearances)).all():
        raise ValueError("SVM frequencies use an incorrect donor denominator.")
    for direction in ("positive", "negative"):
        np.testing.assert_allclose(svm[f"{direction}_frequency"],
                                   svm[f"{direction}_count"] / svm.n_test_models, rtol=0, atol=1e-12)
    # This existing helper independently checks summary quantiles against split metrics.
    for pairwise in (False, True):
        metric_table(tables, pairwise=pairwise)
    return dict(**frames, reference=reference, exploration=exploration, provenance=provenance)


def aggregate_svm_profiles(records, markers=MARKERS):
    """Mean cells per visit, then nonempty visits per donor, then equal donors."""
    if records.duplicated(["split_id", "donor_id"]).any():
        raise ValueError("Duplicate SVM donor/test visit.")
    if not records.n_selected.ge(0).all():
        raise ValueError("Negative selected-cell count.")
    nonempty = records.loc[records.n_selected.gt(0)]
    if nonempty.empty:
        raise ValueError("SVM: keine positiv ausgewählten Testzellen; kein Markerprofil bestimmbar.")
    if (not np.isfinite(nonempty[list(markers)].to_numpy()).all()
            or not records.loc[records.n_selected.eq(0), list(markers)].isna().all().all()):
        raise ValueError("SVM profiles must be finite for nonempty visits and missing for empty visits.")
    donor_profiles = nonempty.groupby("donor_id")[list(markers)].mean()
    coverage = dict(test_visits=len(records), nonempty_test_visits=len(nonempty),
                    test_donors=records.donor_id.nunique(), represented_donors=len(donor_profiles),
                    selected_cell_occurrences=int(records.n_selected.sum()))
    return donor_profiles.mean().to_numpy(), coverage


def svm_profile_records(inputs, models, predictions, splits, saved_cells, markers=MARKERS):
    """Reconstruct full-donor selections and check scores plus saved map counts."""
    marker_indices = [inputs["markers"].index(marker) for marker in markers]
    cells = inputs["cells"]
    saved_cells = align_cells(saved_cells, cells)
    counts = np.zeros((len(cells), 3), dtype=int)
    predictions = predictions.set_index(["split_id", "donor_id"])
    records = []
    for split_id, split in splits.groupby("split_id", sort=True):
        parameters = models.loc[models.split_id.eq(split_id)].set_index("marker").loc[inputs["markers"]]
        scaler = restore_scaler(parameters)
        threshold = float(parameters.decision_threshold.iloc[0])
        for donor in sorted(split.loc[split.outer_partition.eq("test"), "donor_id"]):
            values = inputs["data"][donor]
            margins = scaler.transform(values) @ parameters.weight.to_numpy() + parameters.intercept.iloc[0]
            positive, negative, top = svm_cell_selection(margins, threshold)
            saved = predictions.loc[(split_id, donor)]
            score_error = abs(margins[top].mean() - saved.score)
            if (score_error > 1e-10 or threshold != saved.decision_threshold
                    or len(top) != saved.top_cell_count):
                raise ValueError(f"SVM prediction mismatch: split {split_id}, donor {donor}.")
            n_selected = int(positive.sum())
            profile = (values[np.ix_(positive, marker_indices)].mean(axis=0, dtype=np.float64)
                       if n_selected else np.full(len(markers), np.nan))
            records.append(dict(split_id=split_id, donor_id=donor, n_cells=len(values),
                                n_top=len(top), n_selected=n_selected, score_error=score_error,
                                **dict(zip(markers, profile))))
            positions = np.flatnonzero(cells.sample_id.eq(donor))
            events = cells.iloc[positions].event_index.to_numpy()
            counts[positions] += np.column_stack((positive[events], negative[events], np.ones(len(events), dtype=int)))
    np.testing.assert_array_equal(counts, saved_cells[["positive_count", "negative_count", "n_test_models"]])
    return pd.DataFrame(records)


def svm_profiles(root, results):
    """Use checked original float32 inputs and frozen outer models, without fitting."""
    inputs = load_inputs(root)
    if inputs["input_hashes"] != results["provenance"]["input_sha256"]:
        raise ValueError("SVM marker profile inputs disagree with task 5 provenance.")
    tables = root / "results/tables"
    splits = inputs["splits"].loc[inputs["splits"].split_id.lt(30)]
    models = csv_read(tables / "task4_svm_models_gated_alive_full.csv")
    predictions = csv_read(tables / "task4_svm_predictions_gated_alive_full.csv")
    models = models.loc[models.split_id.lt(30)]
    predictions = predictions.loc[predictions.split_id.lt(30)]
    validate_prediction_splits(predictions, splits)
    validate_parameter_table(models, predictions, inputs["markers"], ["split_id"])
    records = svm_profile_records(inputs, models, predictions, splits, results["svm_cells"])
    raw, coverage = aggregate_svm_profiles(records)
    return dict(records=records, raw=raw, coverage=coverage, input_sha256=inputs["input_hashes"])


def save(fig, output, name):
    fig.savefig(output / f"{name}.pdf", dpi=250, bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def performance(tables, figures, data, *, pairwise=False):
    prefix = "task4_cellcnn_svm_100" if pairwise else "task4"
    metrics = pd.read_csv(tables / f"{prefix}_split_metrics.csv")
    methods = (0, 2) if pairwise else (0, 1, 2)
    fig, ax = plt.subplots(figsize=(6.7, 3.5), layout="constrained")
    rng = np.random.default_rng(45)  # Display-only jitter; never used for inference.
    rows = []
    for position, index in enumerate(methods, 1):
        values = metrics.loc[metrics.method_label.eq(METHODS[index])].sort_values("split_id")
        ax.boxplot(values.roc_auc, positions=[position], widths=.48, patch_artist=True,
                   showfliers=False, boxprops={"facecolor": COLORS[index], "alpha": .18},
                   medianprops={"color": COLORS[index], "linewidth": 2},
                   whiskerprops={"color": "#526579"}, capprops={"color": "#526579"})
        ax.scatter(position + rng.uniform(-.16, .16, len(values)), values.roc_auc,
                   s=14 if pairwise else 21, color=COLORS[index], alpha=.65, linewidths=0)
        medians = values[["roc_auc", "average_precision", "balanced_accuracy"]].median()
        rows.append(LABELS[index] + " & " + " & ".join(f"{x:.3f}" for x in medians) + r" \\")
    ax.axhline(.5, color="#526579", linewidth=.8, linestyle="--", zorder=0)
    ax.set(ylim=(-.035, 1.045), ylabel="Test ROC-AUC", xticks=range(1, len(methods) + 1),
           xticklabels=[LABELS[i] for i in methods], yticks=np.arange(0, 1.01, .25))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#EFF4F7")
    ax.set_axisbelow(True)
    name = "performance_100" if pairwise else "performance_30"
    save(fig, figures, name)
    table = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Method & AUC & AP & BA \\", r"\midrule", *rows,
             r"\bottomrule", r"\end{tabular}", ""]
    (data / f"{name}.tex").write_text("\n".join(table))


def map_background(ax, reference, title):
    ax.scatter(reference.component_1, reference.component_2, s=2, c="#DBE0E5",
               linewidths=0, rasterized=True)
    ax.set(title=title, aspect="equal", xticks=[], yticks=[])
    ax.title.set_fontsize(14)
    for spine in ax.spines.values():
        spine.set_edgecolor("#DBE4EB")


def centroid_maps(results, figures):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), layout="constrained")
    for ax, method in zip(axes, ("CellCNN", "Citrus")):
        map_background(ax, results["reference"], method)
        for group in results["groups"].loc[lambda x: x.method.eq(method) & x.retained].itertuples():
            members = results["centroids"].loc[lambda x: x.method.eq(method) & x.group_id.eq(group.group_id)]
            color = plt.get_cmap("tab10")((group.group_id - 1) % 10)
            ax.scatter(members.component_1, members.component_2, s=27, color=color,
                       edgecolors="white", linewidths=.4, label=f"G{group.group_id}: {group.occurrences}/30")
            point = members.loc[members.split_id.eq(group.representative_split_id) &
                                members.subset_id.eq(group.representative_subset_id)].iloc[0]
            ax.scatter(point.component_1, point.component_2, s=105, marker="*", color=color,
                       edgecolors="#172B43", linewidths=.65)
        ax.legend(loc="upper left", fontsize=12, handletextpad=.3, borderpad=.4)
    save(fig, figures, "centroids")


def selected_cells(results, figures):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), layout="constrained")
    for ax, title in zip(axes, ("CellCNN representative", "SVM: held-out donors")):
        map_background(ax, results["reference"], title)
    representative = results["representative_cells"]
    selected = representative.loc[representative.selected]
    axes[0].scatter(selected.component_1, selected.component_2, s=17, color=COLORS[0], linewidths=0)
    svm = results["svm_cells"].loc[lambda x: x.positive_frequency.gt(0)].sort_values("positive_frequency")
    points = axes[1].scatter(svm.component_1, svm.component_2, c=svm.positive_frequency,
                            vmin=0, vmax=1, cmap="viridis", s=17, linewidths=0)
    fig.colorbar(points, ax=axes[1], ticks=[0, .5, 1], shrink=.7, pad=.02,
                 label="Positive selection frequency")
    save(fig, figures, "selected_cells")


def marker_profiles(results, figures, data):
    exploration = results["exploration"]
    indices = [exploration["markers"].index(marker) for marker in MARKERS]
    means, scales = np.array(exploration["scaler_mean"])[indices], np.array(exploration["scaler_scale"])[indices]
    rows, values = [], []
    for method in ("CellCNN", "Citrus"):
        group = results["groups"].loc[lambda x: x.method.eq(method) & x.retained].sort_values("group_id").iloc[0]
        point = results["centroids"].loc[lambda x: x.method.eq(method) &
            x.split_id.eq(group.representative_split_id) & x.subset_id.eq(group.representative_subset_id)].iloc[0]
        raw = point[list(MARKERS)].to_numpy(dtype=float)
        z = (raw - means) / scales
        values.append(z)
        rows.extend(dict(method=method, profile_kind="training_representative",
                         group_id=int(group.group_id), split_id=int(point.split_id),
                         subset_id=int(point.subset_id), marker=marker, arcsinh_value=a, z_value=b)
                    for marker, a, b in zip(MARKERS, raw, z))
    raw = results["svm_profile"]["raw"]
    z = (raw - means) / scales
    values.append(z)
    rows.extend(dict(method="SVM", profile_kind="donor_balanced_positive_oof", group_id=None,
                     split_id=None, subset_id=None, marker=marker, arcsinh_value=a, z_value=b)
                for marker, a, b in zip(MARKERS, raw, z))
    pd.DataFrame(rows).to_csv(data / "representative_profiles.csv", index=False)
    values = np.array(values)
    limit = max(1, np.ceil(np.abs(values).max()))
    fig, ax = plt.subplots(figsize=(7.4, 2.2), layout="constrained")
    hm = ax.imshow(values, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set(xticks=range(len(MARKERS)), xticklabels=MARKERS, yticks=[0, 1, 2],
           yticklabels=["CellCNN G1", "Citrus G1", "SVM positive OOF"])
    ax.tick_params(length=0, pad=7)
    for row in range(3):
        for col in range(len(MARKERS)):
            value = values[row, col]
            label = f"{value:.1f}" if abs(value) >= .05 else "0.0"
            ax.text(col, row, label, ha="center", va="center",
                    color="white" if abs(value) > .55 * limit else "#172B43", fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(hm, ax=ax, ticks=[-limit, 0, limit], pad=.02, label="Exploratory z-scale")
    save(fig, figures, "marker_profiles")


def main():
    tables = ROOT / "results/tables"
    results = load_results(tables)  # Fail before exporting if source results are inconsistent.
    results["svm_profile"] = svm_profiles(ROOT, results)
    figures, data = OUTPUT / "figures", OUTPUT / "data"
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    results["svm_profile"]["records"].to_csv(data / "svm_profiles_by_test_visit.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "#172B43", "axes.labelcolor": "#172B43",
                         "xtick.color": "#172B43", "ytick.color": "#172B43", "pdf.fonttype": 42})
    performance(tables, figures, data)
    performance(tables, figures, data, pairwise=True)
    centroid_maps(results, figures)
    selected_cells(results, figures)
    marker_profiles(results, figures, data)
    groups, svm, rep = results["groups"], results["svm_cells"], results["representative_cells"]
    macros = {"CNNRecurrence": int(groups.loc[groups.method.eq("CellCNN"), "occurrences"].max()),
              "CitrusRecurrence": int(groups.loc[groups.method.eq("Citrus"), "occurrences"].max()),
              "CNNSelected": int(rep.selected.sum()), "SVMSelected": int(svm.positive_count.gt(0).sum()),
              "MinTestModels": int(svm.n_test_models.min()), "MaxTestModels": int(svm.n_test_models.max()),
              "SVMProfileDonors": results["svm_profile"]["coverage"]["represented_donors"],
              "SVMProfileVisits": results["svm_profile"]["coverage"]["nonempty_test_visits"]}
    (data / "numbers.tex").write_text("".join("\\newcommand{\\" + key + "}{" + str(value) + "}\n"
                                             for key, value in macros.items()))
    names = ["task5_paper_provenance.json", *results["provenance"]["artifact_sha256"],
             *results["provenance"]["output_sha256"], "task4_metric_summary.csv", "task4_split_metrics.csv",
             "task4_cellcnn_svm_100_metric_summary.csv", "task4_cellcnn_svm_100_split_metrics.csv"]
    sources = {name: file_hash(tables / name) for name in sorted(set(names))}
    generated = sorted(figures.glob("*.pdf")) + sorted(data.glob("*.tex")) + [
        data / "representative_profiles.csv", data / "svm_profiles_by_test_visit.csv"]
    (data / "provenance.json").write_text(json.dumps(dict(
        gate="gated_alive", main_split_ids=list(range(30)), reserve_split_ids=list(range(100)),
        map="task2 tsne_p30; 500 cells/donor; 10000 cells; no refit",
        representative="stored G1 medoids; no reselection", markers=list(MARKERS),
        svm_profile=dict(aggregation="mean selected ArcSinh cells per test visit; mean nonempty visits per donor; equal donor mean",
                         selection="full test donor top ceil(1%); margin > saved threshold; ties by event index",
                         empty_selection="missing profile; excluded from profile means, retained in coverage",
                         arithmetic="original float32 ArcSinh/scaler inference; float64 profile accumulation",
                         coverage=results["svm_profile"]["coverage"],
                         input_sha256=results["svm_profile"]["input_sha256"]),
        display_jitter_seed=45, source_sha256=sources, exporter_sha256=file_hash(Path(__file__)),
        output_sha256={str(p.relative_to(OUTPUT)): file_hash(p) for p in generated}), indent=2) + "\n")
    print("Exportiert: fünf PDF-Grafiken, zwei Median-Tabellen, Markerwerte und Provenienz.")
    print("Geprüft: 30/100-Split-Quantile, Aufgabe-5-Quellen, Karten-IDs und Häufigkeitsnenner.")
    print("SVM-Profil:", results["svm_profile"]["coverage"])


if __name__ == "__main__":
    main()
