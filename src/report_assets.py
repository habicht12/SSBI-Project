"""Export the report's figures and table from saved analyses, without fitting.

Run from the project root: python -m src.report_assets
Only the CD3 colours require read-only access to the original FCS files.
"""

from pathlib import Path
import hashlib
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("CellCNN", "Citrus", "SVM")
COLOURS = {"CellCNN": "#0072B2", "Citrus": "#E69F00", "SVM": "#009E73"}
LABELS = {"CellCNN": "CellCNN", "Citrus": "Citrus", "SVM": "Linear SVM"}
MARKERS = ["CD3", "CD4", "CD8", "CD19", "CD33", "CD11b", "CD56", "CD16",
           "CD94", "NKG2A", "NKG2C", "CD57"]
VARIANTS = ["pca", "pca_white", "tsne_p5", "tsne_p30", "tsne_p50",
            "umap_n15_d0.1", "umap_n50_d0.1", "umap_n15_d0.5"]
SHORT_NAMES = ["P", "Pw", "T5", "T30", "T50", "U15", "U50", "Ud"]
WIDTH = 17 / 2.54


def align_cells(frame, cells):
    """Require identical cell identities and restore the canonical row order."""
    if frame.cell_id.duplicated().any() or cells.cell_id.duplicated().any():
        raise ValueError("Duplicate cell IDs in report inputs.")
    if set(frame.cell_id) != set(cells.cell_id):
        raise ValueError("Report inputs do not contain the same cells.")
    aligned = frame.set_index("cell_id").loc[cells.cell_id].reset_index()
    for column in ("sample_id", "event_index"):
        if column in aligned and not np.array_equal(aligned[column], cells[column]):
            raise ValueError(f"Inconsistent {column} for report cells.")
    return aligned


def load_cd3(root, cells, provenance):
    """Read precisely the saved events; check the original exploration digest."""
    import flowkit as fk

    if provenance["gate"] != "gated_alive" or len(cells) != 10_000:
        raise ValueError("Expected the 10,000-cell gated_alive exploration.")
    markers = provenance["markers"]
    raw = np.empty((len(cells), len(markers)))
    for donor, group in cells.groupby("sample_id", sort=False):
        path = root / "NK_cell_dataset/NK_cell_dataset/NK_cell_dataset/gated_alive" / f"{donor}_alive.fcs"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"FCS file .* reported incorrect data offset.*")
            sample = fk.Sample(str(path), ignore_offset_error=True)
        columns = [sample.pns_labels.index(marker) for marker in markers]
        raw[group.index] = sample.get_events(source="raw")[np.ix_(group.event_index, columns)]
    digest = hashlib.sha256()
    digest.update(json.dumps({"cell_ids": cells.cell_id.tolist(), "markers": markers},
                             separators=(",", ":")).encode())
    digest.update(raw.astype("<f8").tobytes())
    if digest.hexdigest() != provenance["selected_data_sha256"]:
        raise ValueError("Original events do not match the saved exploration.")
    return np.arcsinh(raw[:, markers.index("CD3")] / 5)


def save_figure(fig, path):
    # Fixed physical width preserves the minimum 8 pt labels in the report.
    fig.savefig(path, dpi=300, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def projection_figure(tables, cells, cd3, output):
    embeddings = pd.read_csv(tables / "task2_embeddings.csv")
    pairs = pd.read_csv(tables / "task2_pairwise_jaccard.csv")
    if len(pairs) != 28 or pairs[["first", "second"]].duplicated().any():
        raise ValueError("Expected all 28 distinct projection pairs.")
    matrix = pd.DataFrame(np.eye(8), index=VARIANTS, columns=VARIANTS)
    seen = set()
    for row in pairs.itertuples():
        pair = frozenset((row.first, row.second))
        if len(pair) != 2 or pair in seen or not 0 <= row.neighbor_jaccard <= 1:
            raise ValueError("Invalid projection comparison.")
        seen.add(pair)
        matrix.loc[row.first, row.second] = matrix.loc[row.second, row.first] = row.neighbor_jaccard
    fig = plt.figure(figsize=(WIDTH, 2.65))
    grid = fig.add_gridspec(2, 3, left=.018, right=.96, bottom=.20, top=.90,
                           width_ratios=[1, 1, 2.12], wspace=.16, hspace=.35)
    order = np.random.default_rng(42).permutation(len(cells))
    norm = Normalize(cd3.min(), cd3.max())
    configurations = [("pca", "A  PCA"), ("tsne_p5", "B  t-SNE · p = 5"),
                      ("umap_n15_d0.1", "C  UMAP · n = 15"), ("tsne_p30", "D  t-SNE · p = 30")]
    for (variant, title), (row, col) in zip(configurations, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        ax = fig.add_subplot(grid[row, col])
        frame = align_cells(embeddings.loc[embeddings.variant.eq(variant)], cells)
        scatter = ax.scatter(frame.component_1.iloc[order], frame.component_2.iloc[order],
                             c=cd3[order], s=.65, cmap="viridis", norm=norm,
                             linewidths=0, rasterized=True)
        ax.set_title(title, loc="left", fontsize=8.5, pad=4)
        ax.set(aspect="equal", xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(.5)
    bar_ax = fig.add_axes([.055, .13, .37, .018])
    fig.colorbar(scatter, cax=bar_ax, orientation="horizontal")
    fig.text(.24, .018, "CD3: arcsinh(intensity / 5)", ha="center", fontsize=8)
    ax = fig.add_subplot(grid[:, 2])
    annotations = np.array([["1" if i == j else f"{matrix.iloc[i, j]:.2f}".lstrip("0")
                             for j in range(8)] for i in range(8)])
    sns.heatmap(matrix, ax=ax, cmap="viridis", vmin=0, vmax=1, annot=annotations, fmt="",
                annot_kws={"fontsize": 8}, square=True, cbar=False,
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES)
    ax.set_title("E  Neighbour agreement (Jaccard)", loc="left", fontsize=8.5, pad=7)
    ax.tick_params(axis="both", length=0, labelrotation=0)
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    save_figure(fig, output / "projections.pdf")


def clustering_figure(tables, output):
    profiles = pd.read_csv(tables / "task3_marker_profiles.csv")
    recommendations = pd.read_csv(tables / "task3_recommendations.csv")
    provenance = json.loads((tables / "task3_provenance.json").read_text())
    columns = provenance["markers"]
    # Apply the saved exploration scale to existing profiles, without fitting.
    profiles[columns] = (profiles[columns] - np.array(provenance["scaler_mean"])) / np.array(provenance["scaler_scale"])
    limit = max(1., np.abs(profiles[MARKERS].to_numpy()).max())
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.75))
    fig.subplots_adjust(left=.035, right=.91, bottom=.19, top=.83, wspace=.27)
    cax = fig.add_axes([.93, .19, .017, .64])
    for letter, ax, method in zip("ABC", axes, ["K-Means", "Ward", "Leiden"]):
        row = recommendations.loc[recommendations.method.eq(method)].iloc[0]
        matrix = profiles.loc[profiles.variant.eq(row.variant)].set_index("cluster")[MARKERS]
        sns.heatmap(matrix, ax=ax, cmap="vlag", center=0, vmin=-limit, vmax=limit,
                    cbar=ax is axes[-1], cbar_ax=cax if ax is axes[-1] else None,
                    xticklabels=True, yticklabels=True)
        ax.set_title(f"{letter}  {method} · {row.n_clusters} clusters\nSilhouette {row.silhouette:.3f}",
                     loc="left", fontsize=9, pad=7)
        ax.set(xlabel="", ylabel="")
        ax.tick_params(axis="both", length=0)
        ax.tick_params(axis="x", labelrotation=90)
        ax.tick_params(axis="y", labelrotation=0)
    cax.set_title("z", fontsize=8, pad=5)
    save_figure(fig, output / "clusters.pdf")


def validate_interpretation(tables):
    """Nur aktuelle 30-Split-Ergebnisse mit unveränderten Quellen exportieren."""
    provenance = json.loads((tables / "task5_paper_provenance.json").read_text())
    if provenance["gate"] != "gated_alive" or provenance["split_ids"] != list(range(30)):
        raise ValueError("Aufgabe 5 benötigt den gemeinsamen Vergleich über 30 Splits.")
    for key in ("artifact_sha256", "output_sha256"):
        for name, expected in provenance[key].items():
            with (tables / name).open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Aufgabe-5-Quelle oder Ergebnis verändert: {name}.")
    implementation = Path(__file__).with_name("task5_interpretation.py")
    if hashlib.sha256(implementation.read_bytes()).hexdigest() != provenance["implementation_sha256"]:
        raise ValueError("Aufgabe-5-Code wurde seit dem Interpretationslauf verändert.")
    return provenance


def interpretation_figure(tables, cells, output):
    centroids = pd.read_csv(tables / "task5_paper_centroids.csv")
    groups = pd.read_csv(tables / "task5_paper_groups.csv")
    scores = align_cells(pd.read_csv(tables / "task5_paper_svm_cells.csv"), cells)
    embeddings = pd.read_csv(tables / "task2_embeddings.csv")
    reference = align_cells(embeddings.loc[embeddings.variant.eq("tsne_p30")], cells)
    if not np.allclose(scores[["component_1", "component_2"]],
                       reference[["component_1", "component_2"]], rtol=0, atol=1e-12):
        raise ValueError("SVM-Koordinaten passen nicht zur gespeicherten Karte.")
    mapped = reference.set_index("cell_id").loc[centroids.map_cell_id]
    if not np.allclose(centroids[["component_1", "component_2"]],
                       mapped[["component_1", "component_2"]], rtol=0, atol=1e-12):
        raise ValueError("Zentroidprojektion passt nicht zu den zugeordneten Karten-Zellen.")
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.75), layout="constrained")
    for ax, title in zip(axes, ["A  CellCNN-Zentroiden", "B  Citrus-Zentroiden", "C  SVM: positive Auswahl"]):
        ax.scatter(reference.component_1, reference.component_2, s=1, c="#dddddd",
                   linewidths=0, rasterized=True)
        ax.set(title=title, aspect="equal", xticks=[], yticks=[])
        ax.title.set_fontsize(8)
    for ax, method in zip(axes[:2], ["CellCNN", "Citrus"]):
        retained = groups.loc[groups.method.eq(method) & groups.retained]
        for group in retained.itertuples():
            frame = centroids.loc[centroids.method.eq(method) & centroids.group_id.eq(group.group_id)]
            color = plt.get_cmap("tab10")((group.group_id - 1) % 10)
            ax.scatter(frame.component_1, frame.component_2, s=12, color=color,
                       edgecolors="white", linewidths=.3, label=f"G{group.group_id}: {group.occurrences}/30")
            point = frame.loc[frame.split_id.eq(group.representative_split_id) &
                              frame.subset_id.eq(group.representative_subset_id)].iloc[0]
            ax.scatter(point.component_1, point.component_2, s=45, marker="*", color=color,
                       edgecolors="black", linewidths=.4)
        if retained.empty:
            ax.text(.5, .5, "Keine Gruppe ab 6/30", transform=ax.transAxes, ha="center", fontsize=8)
        else:
            ax.legend(fontsize=8, loc="upper left", handletextpad=.2, borderpad=.3, labelspacing=.2)
    if not scores.n_test_models.gt(0).all() or not np.allclose(
            scores.positive_frequency, scores.positive_count / scores.n_test_models):
        raise ValueError("Ungültiger Nenner der SVM-Auswahlhäufigkeit.")
    chosen = scores.loc[scores.positive_frequency.gt(0)].sort_values("positive_frequency")
    points = axes[2].scatter(chosen.component_1, chosen.component_2, c=chosen.positive_frequency,
                            cmap="viridis", vmin=0, vmax=1, s=3, linewidths=0, rasterized=True)
    fig.colorbar(points, ax=axes[2], ticks=[0, .5, 1], shrink=.6, label="SVM: OOF-Häufigkeit")
    save_figure(fig, output / "interpretation.pdf")


def metric_table(tables, *, pairwise=False):
    prefix = "task4_cellcnn_svm_100" if pairwise else "task4"
    split_count = 100 if pairwise else 30
    methods = ("CellCNN", "SVM") if pairwise else METHODS
    summary = pd.read_csv(tables / f"{prefix}_metric_summary.csv")
    split_metrics = pd.read_csv(tables / f"{prefix}_split_metrics.csv")
    lines = [f"% {split_count} gemeinsame Splits; Median [Q1, Q3].",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccc@{}}", r"\toprule",
             r"Methode & ROC-AUC & Average Precision & Balanced Accuracy \\", r"\midrule"]
    for method in methods:
        source_name = "Lineare Single-Cell-SVM" if method == "SVM" else method
        values = []
        subset = split_metrics.loc[split_metrics.method_label.eq(source_name)]
        if len(subset) != split_count or set(subset.split_id) != set(range(split_count)):
            raise ValueError(f"Expected {split_count} common benchmark splits per method.")
        for metric in ("roc_auc", "average_precision", "balanced_accuracy"):
            row = summary.loc[summary.Methode.eq(source_name) & summary.Metrik.eq(metric)]
            if len(row) != 1:
                raise ValueError("Missing or duplicate benchmark summary.")
            median, q1, q3 = row[["Median", "Q1", "Q3"]].iloc[0]
            if not np.allclose([median, q1, q3], subset[metric].quantile([.5, .25, .75])):
                raise ValueError("Benchmark summary disagrees with split metrics.")
            values.append(f"{median:.3f} [{q1:.3f}, {q3:.3f}]")
        lines.append(r"\textcolor{" + method.lower() + "}{" + LABELS[method] + "} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", ""])
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gespeicherte Berichtsassets exportieren.")
    parser.add_argument("--classification-only", action="store_true",
                        help="Nur Aufgabe-4-Tabellen; Aufgabe 5 bleibt unverändert.")
    args = parser.parse_args()
    tables = ROOT / "results/tables"
    if args.classification_only:
        table_dir = ROOT / "report/tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        for filename, pairwise in [("classification.tex", False), ("classification_cellcnn_svm_100.tex", True)]:
            (table_dir / filename).write_text(metric_table(tables, pairwise=pairwise))
        print("Aufgabe-4-Tabellen für 30 bzw. 100 gemeinsame Splits aktualisiert.")
        return
    cells = pd.read_csv(tables / "task2_cells.csv")
    provenance = json.loads((tables / "task2_provenance.json").read_text())
    profile_provenance = json.loads((tables / "task3_provenance.json").read_text())
    interpretation_provenance = validate_interpretation(tables)
    if len({p["selected_data_sha256"] for p in [provenance, profile_provenance, interpretation_provenance]}) != 1:
        raise ValueError("Exploration, clustering and interpretation use different cells.")
    align_cells(cells, cells)
    cd3 = load_cd3(ROOT, cells, provenance)
    latex = metric_table(tables)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "text.color": "#222222", "axes.labelcolor": "#333333",
                         "pdf.fonttype": 42, "axes.spines.top": True, "axes.spines.right": True})
    output = ROOT / "report/figures"
    output.mkdir(parents=True, exist_ok=True)
    projection_figure(tables, cells, cd3, output)
    clustering_figure(tables, output)
    interpretation_figure(tables, cells, output)
    table_dir = ROOT / "report/tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "classification.tex").write_text(latex)
    (table_dir / "classification_cellcnn_svm_100.tex").write_text(metric_table(tables, pairwise=True))
    print("Exported three report figures and the checked classification table; no models were fitted.")


if __name__ == "__main__":
    main()
