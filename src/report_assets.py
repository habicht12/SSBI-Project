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


def interpretation_figure(tables, cells, output):
    scores = pd.read_csv(tables / "task5_cell_scores.csv")
    embeddings = pd.read_csv(tables / "task2_embeddings.csv")
    reference = align_cells(embeddings.loc[embeddings.variant.eq("tsne_p30")], cells)
    profiles = pd.read_csv(tables / "task5_report_marker_table.csv").set_index("method")
    fig = plt.figure(figsize=(WIDTH, 3.15))
    grid = fig.add_gridspec(2, 3, left=.018, right=.915, top=.90, bottom=.03,
                           height_ratios=[2, 1.04], wspace=.11, hspace=.15)
    for letter, column, method in zip("ABC", range(3), METHODS):
        ax = fig.add_subplot(grid[0, column])
        frame = align_cells(scores.loc[scores.method.eq(method)], cells)
        if not np.allclose(frame[["component_1", "component_2"]],
                           reference[["component_1", "component_2"]], rtol=0, atol=1e-12):
            raise ValueError("Interpretation coordinates differ from the saved t-SNE map.")
        if not frame.positive_frequency.between(0, 1).all() or not frame.n_test_models.gt(0).all():
            raise ValueError("Invalid held-out selection frequencies.")
        if not np.allclose(frame.positive_frequency, frame.positive_count / frame.n_test_models):
            raise ValueError("Selection frequency does not match its denominator.")
        ax.scatter(frame.component_1, frame.component_2, s=1, c="#dddddd", linewidths=0, rasterized=True)
        chosen = frame.loc[frame.positive_frequency.gt(0)].sort_values("positive_frequency")
        points = ax.scatter(chosen.component_1, chosen.component_2, c=chosen.positive_frequency,
                            cmap="viridis", vmin=0, vmax=1, s=2, linewidths=0, rasterized=True)
        ax.set_title(f"{letter}  {LABELS[method]}", loc="left", color=COLOURS[method], fontsize=9.5, pad=7)
        ax.set(aspect="equal", xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(.5)
    cax = fig.add_axes([.938, .42, .017, .45])
    fig.colorbar(points, cax=cax, ticks=[0, .5, 1])
    cax.set_title("f⁺", fontsize=9)
    ax = fig.add_subplot(grid[1, :])
    ax.axis("off")
    rows = []
    for method in [*METHODS, "Kartenreferenz"]:
        rows.append([LABELS.get(method, "Map reference"),
                     *[f"{profiles.loc[method, marker]:.2f}" for marker in ["CD3", "CD56", "NKG2C", "CD57"]]])
    table = ax.table(cellText=rows, colLabels=["Positive selection", "CD3", "CD56", "NKG2C", "CD57"],
                     colWidths=[.32, .17, .17, .17, .17], cellLoc="center", bbox=[.075, 0, .86, .93])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0)
        if row == 0:
            cell.set_facecolor("#f0f2f4")
            cell.set_text_props(weight="bold")
        elif col == 0:
            cell.set_text_props(color=COLOURS[METHODS[row - 1]] if row <= 3 else "#777777")
    save_figure(fig, output / "interpretation.pdf")


def metric_table(tables):
    summary = pd.read_csv(tables / "task4_metric_summary.csv")
    split_metrics = pd.read_csv(tables / "task4_split_metrics.csv")
    lines = [r"% Generated by python -m src.report_assets; median [Q1, Q3].",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lccc@{}}", r"\toprule",
             r"Method & ROC-AUC & Average precision & Balanced accuracy \\", r"\midrule"]
    for method in METHODS:
        source_name = "Lineare Single-Cell-SVM" if method == "SVM" else method
        values = []
        subset = split_metrics.loc[split_metrics.method_label.eq(source_name)]
        if len(subset) != 100 or set(subset.split_id) != set(range(100)):
            raise ValueError("Expected 100 full benchmark splits per method.")
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
    tables = ROOT / "results/tables"
    cells = pd.read_csv(tables / "task2_cells.csv")
    provenance = json.loads((tables / "task2_provenance.json").read_text())
    profile_provenance = json.loads((tables / "task3_provenance.json").read_text())
    interpretation_provenance = json.loads((tables / "task5_provenance.json").read_text())
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
    print("Exported three report figures and the checked classification table; no models were fitted.")


if __name__ == "__main__":
    main()
