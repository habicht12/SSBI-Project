"""Export saved notebook outputs for the slides, without rerunning clustering."""
from pathlib import Path
from io import StringIO
import base64
import hashlib
import json
import os
import re

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE.parent / "notebooks/03_clustering.ipynb"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ssbi-presentation-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    nb = json.loads(NOTEBOOK.read_text())
    figures = HERE / "figures"
    data = HERE / "data"
    figures.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    tables = []
    umaps = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is not None, "Execute the entire notebook first."
        for output in cell.get("outputs", []):
            assert output["output_type"] != "error", "Notebook contains an execution error."
            content = output.get("data", {})
            if "text/html" in content:
                tables.extend(pd.read_html(StringIO("".join(content["text/html"])), index_col=0))
            if "image/png" in content and 'embedding = plot_data.obsm["X_umap"]' in "".join(cell["source"]):
                umaps.append(base64.b64decode("".join(content["image/png"])))

    grid = next(t for t in tables if {"eligible", "stability", "id"} <= set(t.columns))
    clusters = next(t for t in tables if {"median_jaccard", "original_cells"} <= set(t.columns))
    markers = next(t for t in tables if "higher_markers (IQR units)" in t.columns)
    results = next(t for t in tables if "Selected parameters" in t.columns)
    assert len(grid) == 44 and len(results) == len(umaps) == 6, "Review the slide content for changed settings."
    assert clusters.groupby("method").original_cells.sum().eq(20000).all()
    assert markers.groupby("method").cells.sum().eq(20000).all()
    for name, table in [("grid", grid), ("selected_cluster_stability", clusters),
                        ("marker_profiles", markers), ("winners", results)]:
        table.to_csv(data / f"{name}.csv", index=False)
    methods = ["k-means", "Hierarchical (single)", "Hierarchical (average)",
               "Hierarchical (complete)", "Hierarchical (ward)", "Leiden"]
    for method, filename in [("k-means", "umap_kmeans.png"), ("Leiden", "umap_leiden.png")]:
        (figures / filename).write_bytes(umaps[methods.index(method)])

    # The notebook stores these medians to three decimal places; no re-estimation.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10.8, 2.35), layout="constrained")
    colors = {"k-means": "#087F8C", "Hierarchical (ward)": "#DD8D29", "Leiden": "#4956A3"}
    for y, method in enumerate(colors):
        values = clusters[clusters.method == method].sort_values("cluster").median_jaccard.to_numpy()
        jitter = np.linspace(-.14, .14, len(values))
        ax.scatter(values, y + jitter, s=75, color=colors[method], edgecolor="white", linewidth=.6, zorder=3)
        weak = int((values < .6).sum())
        ax.text(1.035, y, f"{weak}/{len(values)} below 0.6", va="center", fontsize=11, color="#425466")
    ax.axvline(.6, color="#AF4D35", linestyle="--", linewidth=1.4)
    ax.set(yticks=[0, 1, 2], yticklabels=["K-means", "Ward", "Leiden"],
           xticks=np.arange(0, 1.01, .2), xlim=(-.03, 1.29), ylim=(2.5, -.5),
           xlabel="Median Jaccard overlap per cluster (10 repeats)")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E5EAF0", linewidth=.8)
    ax.set_axisbelow(True)
    fig.savefig(figures / "cluster_stability.pdf", bbox_inches="tight")
    plt.close(fig)

    def number(value):
        return f"{float(value):.3f}"

    rows = []
    for method, name, saved in zip(methods, ["K-means", "Single", "Average", "Complete", "Ward", "Leiden"], results.to_dict("records")):
        # Read the actual saved selection; rounding could change score ordering.
        settings = saved["Selected parameters"]
        if method == "Leiden":
            match = re.fullmatch(r"(\d+) neighbours, resolution=([\d.]+); (\d+) clusters", settings)
            assert match, settings
            chosen = grid[(grid.method == method) & (grid.neighbors == int(match[1])) &
                          (grid.resolution == float(match[2]))]
        else:
            match = re.fullmatch(r"k=(\d+)", settings)
            assert match, settings
            chosen = grid[(grid.method == method) & (grid.k == int(match[1]))]
        assert len(chosen) == 1
        winner = chosen.iloc[0]
        assert winner.eligible
        assert float(saved["davies_bouldin ↓"]) == winner.score
        assert float(saved["Mean stability"]) == winner.stability
        cs = clusters[clusters.method == method]
        n_clusters = len(cs)
        largest = 100 * cs.original_cells.max() / cs.original_cells.sum()
        bs = chr(92)
        parameter = (f"$k={int(winner.k)}$" if method != "Leiden"
                     else f"$n={int(winner.neighbors)}, r={winner.resolution:g}$")
        rows.append(f"{name} & {parameter} & {n_clusters} & {number(winner.score)} & {number(winner.stability)} & {number(largest)}{bs},{bs}% {bs}{bs}")
    table_lines = [bs + "begin{tabular}{@{}llrrrr@{}}", bs + "toprule",
                   f"Method & Parameters & Clusters & DB ${bs}downarrow$ & Stability & Largest cluster{bs}{bs}",
                   bs + "midrule", *rows, bs + "bottomrule", bs + "end{tabular}"]
    (data / "winner_table.tex").write_text(chr(10).join(table_lines) + chr(10))
    manifest = {"source": "notebooks/03_clustering.ipynb", "sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
                "cells": 20000, "markers": 37, "configurations": 44,
                "note": "Values and original UMAP images extracted from saved outputs; numeric tables rounded by notebook."}
    (data / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Exported saved results, two original UMAPs and the cluster-stability figure.")


if __name__ == "__main__":
    main()
