"""Build report figures/tables from verified results; reconstruct only 3 Task-3 winners.

First run: .venv/bin/python report/export_assets.py --reconstruct-clusters
Later:    .venv/bin/python report/export_assets.py
"""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[key] = "2"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ssbi-report-mpl")

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import procrustes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report"
ASSETS = ROOT / "praesentation/report_assets"
PREP = ROOT / "results/tables/report_preparation_20260913"
DISPLAY = ["CD3", "CD4", "CD8", "CD19", "CD33", "CD11b", "CD34", "CD56", "CD94", "NKp46", "NKG2C", "CD57"]
METHODS = ["k-means", "Hierarchical (ward)", "Leiden"]
LABELS = ["K-means", "Ward", "Leiden"]


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def reconstruct_clusters():
    """Execute original loading/scaling cells, then fit only the fixed winners."""
    source = ROOT / "notebooks/03_clustering.ipynb"
    notebook = json.loads(source.read_text())
    env = {"__name__": "report_cluster_reconstruction"}
    exec("".join(notebook["cells"][1]["source"]), env)
    env["ROOT"] = ROOT
    for index in [3, 5]:
        exec("".join(notebook["cells"][index]["source"]), env)
    x, values, markers = env["X"], env["arcsinh_values"], env["markers"]
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score
    from scipy.cluster.hierarchy import linkage, cut_tree
    import scanpy as sc
    print("Reconstruct K-means k=4", flush=True)
    partitions = {METHODS[0]: KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(x)}
    print("Reconstruct Ward k=6 (one 20,000-cell tree)", flush=True)
    partitions[METHODS[1]] = cut_tree(linkage(x, method="ward", metric="euclidean"), n_clusters=[6])[:, 0]
    print("Reconstruct Leiden n=30, resolution=1.2", flush=True)
    data = sc.AnnData(x)
    sc.pp.neighbors(data, n_neighbors=30, use_rep="X", random_state=42)
    sc.tl.leiden(data, resolution=1.2, flavor="leidenalg", directed=False, n_iterations=-1, random_state=42)
    partitions[METHODS[2]] = data.obs.leiden.astype(int).to_numpy()
    old = read(ROOT / "data/marker_profiles.csv")
    print("Saved marker methods:", old.method.unique(), flush=True)
    aliases = {"k-means": "k-means", "Ward": "Hierarchical (ward)", "ward": "Hierarchical (ward)",
               "Ward linkage": "Hierarchical (ward)", "Leiden": "Leiden"}
    old["method"] = old.method.replace(aliases)
    expression = pd.DataFrame(values, columns=markers)
    median, iqr = expression.median(), expression.quantile(.75) - expression.quantile(.25)
    rows, summaries, labels = [], [], []
    for method, expected_clusters, expected_db, stability in zip(METHODS, [4, 6, 18], [2.363, 2.522, 2.361], [.995, .611, .683]):
        partition = partitions[method]
        actual_db = davies_bouldin_score(x, partition)
        if len(np.unique(partition)) != expected_clusters or round(actual_db, 3) != expected_db:
            raise ValueError(f"Reconstructed clustering does not match notebook: {method}, DB={actual_db}")
        profile = expression.groupby(partition).median()
        relative = (profile - median) / iqr.replace(0, np.nan)
        for cluster, row in relative.iterrows():
            count = int((partition == cluster).sum())
            saved = old.loc[old.method.eq(method) & old.cluster.eq(cluster)]
            if len(saved) != 1 or saved.cells.iloc[0] != count:
                raise ValueError(f"Cluster identity/size differs: {method} {cluster}")
            for direction, selected in [("higher", row[row >= .5].nlargest(3)), ("lower", row[row <= -.5].nsmallest(3))]:
                description = ", ".join(f"{m} ({v:+.2f})" for m, v in selected.items()) or "—"
                stored = saved[f"{direction}_markers (IQR units)"].iloc[0]
                agrees = description.startswith(stored[:-3]) if stored.endswith("...") else description == stored
                if not agrees:
                    raise ValueError(f"Marker description differs: {method} {cluster}, {description}")
            for marker in markers:
                rows.append(dict(method=method, cluster=int(cluster), cells=count, marker=marker,
                                 median_arcsinh=profile.loc[cluster, marker], relative_iqr=row[marker]))
        summaries.append(dict(method=method, clusters=expected_clusters, db=actual_db, mean_stability=stability))
        labels.append(env["cell_info"].assign(method=method, cluster=partition))
    pd.DataFrame(rows).to_csv(OUT / "data/cluster_profiles.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "data/cluster_summary.csv", index=False)
    pd.concat(labels).to_csv(OUT / "data/cluster_memberships.csv", index=False)
    (OUT / "data/cluster_reconstruction.json").write_text(json.dumps({
        "source_notebook_sha256": sha(source), "cells": len(x), "gate": "gated_alive",
        "preprocessing": "Original notebook cells 1,3,5; sequential default_rng(42) sampling in donor-file order; arcsinh / 5; StandardScaler",
        "fits": "Only k-means k=4, Ward k=6, Leiden n=30/resolution=1.2, seed 42; no grid or resampling rerun",
        "checks": "Cluster counts, each original cluster size, rounded DB and all visible original top-marker annotations agree; the stored Leiden-16 higher-marker string is truncated, so only its visible prefix is compared",
        "stability": "Original stored resampling results, not newly estimated",
        "output_sha256": {name: sha(OUT / "data" / name) for name in ["cluster_profiles.csv", "cluster_summary.csv", "cluster_memberships.csv"]}
    }, indent=2) + "\n")
    print("Three selected partitions and all marker annotations reproduced.", flush=True)


def save(fig, name):
    fig.savefig(OUT / "figures" / f"{name}.pdf", bbox_inches="tight", dpi=400,
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / "figures" / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def figures():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "text.color": "#23364a", "axes.labelcolor": "#23364a"})
    embeddings_path = PREP / "embeddings_full.parquet"
    embeddings = pd.read_parquet(embeddings_path)
    labels = read(ROOT / "labels.csv").set_index("sample_id").clinical_group
    if not embeddings.clinical_group.eq(embeddings.sample_id.map(labels)).all():
        raise ValueError("Embedding colours do not match the donor labels.")
    original_metadata = pd.read_parquet(ROOT / "nk_balanced_raw.parquet")[["sample_id", "clinical_group"]]
    pd.testing.assert_frame_equal(embeddings[["sample_id", "clinical_group"]].reset_index(drop=True),
                                  original_metadata.reset_index(drop=True), check_dtype=False)
    quality = read(ASSETS / "aufgabe_02/data/quality.csv").set_index("embedding")
    pairs = read(ASSETS / "aufgabe_02/data/pairwise.csv")
    axes_columns = {"PCA": ["PC1", "PC2"], "t-SNE": ["tsne1", "tsne2"], "UMAP": ["umap1", "umap2"]}
    assert len(embeddings) == 40000 and embeddings.groupby("sample_id").size().eq(2000).all()
    for row in pairs.itertuples():
        a, b = row.pair.split(" vs ")
        result = procrustes(embeddings[axes_columns[a]], embeddings[axes_columns[b]])[2]
        np.testing.assert_allclose(result, row.procrustes_disparity, atol=1e-12, rtol=0)
    order = np.random.default_rng(42).permutation(len(embeddings))
    colors = np.where(embeddings.clinical_group.eq("CMV+"), "#d1495b", "#2878b5")
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.35), layout="constrained")
    subtitles = {"PCA": "PC1–PC2", "t-SNE": "perplexity 70", "UMAP": "n=5, min_dist=0, lr=0.1"}
    for ax, (name, columns) in zip(axes, axes_columns.items()):
        points = embeddings[columns].to_numpy()
        ax.scatter(points[order, 0], points[order, 1], c=colors[order], s=1.3, alpha=.5, linewidths=0, rasterized=True)
        ax.set(title=f"{name} · T={quality.loc[name, 'trustworthiness']:.3f}\n{subtitles[name]}", xticks=[], yticks=[], aspect="equal")
        for spine in ax.spines.values(): spine.set_visible(False)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker='o', color='none', markerfacecolor=color, label=name, markersize=4)
                        for name,color in [("CMV−", "#2878b5"), ("CMV+", "#d1495b")]],
               loc="outside upper center", ncol=2, frameon=False, fontsize=7.5)
    fig.supxlabel("Pairwise Procrustes disparity: PCA / t-SNE 0.442 · PCA / UMAP 0.554 · t-SNE / UMAP 0.368", fontsize=7)
    save(fig, "dimensionality_reduction")
    profiles = read(OUT / "data/cluster_profiles.csv")
    summary = read(OUT / "data/cluster_summary.csv").set_index("method")
    for marker_list, filename, height in [(DISPLAY, "clustering", 2.9),
                                           (profiles.marker.unique().tolist(), "clustering_all_markers", 4.7)]:
        if filename == "clustering":
            fig, axes = plt.subplots(1, 3, figsize=(6.4, height), layout="constrained")
        else:
            fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.2), layout="constrained", sharex=True,
                                     gridspec_kw={"height_ratios": [4, 6, 18]})
        for ax, method, label in zip(axes, METHODS, LABELS):
            matrix = profiles.loc[profiles.method.eq(method)].pivot(index="cluster", columns="marker", values="relative_iqr")[marker_list]
            im = ax.imshow(matrix, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
            ax.grid(False)
            ax.set_xticks(range(len(marker_list)), marker_list, rotation=90, fontsize=7)
            ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
            ax.tick_params(length=0)
            row = summary.loc[method]
            ax.set_title(f"{label} · {int(row.clusters)} clusters\nDB={row.db:.3f}; stability={row.mean_stability:.3f}", fontsize=8)
            ax.set_ylabel("Cluster ID", fontsize=7)
            if filename != "clustering" and ax is not axes[-1]:
                ax.tick_params(axis="x", labelbottom=False)
        fig.colorbar(im, ax=list(axes), label="Median difference / sample IQR", ticks=[-2, 0, 2],
                     fraction=.025, pad=.015, shrink=.8)
        save(fig, filename)
    # Existing verified Task 4/5 figures; their data and selection rules are unchanged.
    for old,new in [("task4_performance_paired", "classification"), ("task5_report_combined", "subsets")]:
        for extension in ["pdf", "png"]:
            shutil.copyfile(ASSETS / f"aufgaben_04_05/figures/{old}.{extension}", OUT / f"figures/{new}.{extension}")
    return embeddings_path


def copy_supplement():
    destination = OUT / "figures/supplement"
    destination.mkdir(exist_ok=True)
    source = PREP / "praesentation/report_assets/aufgabe_02"
    for name in ["quality.csv", "pairwise.csv", "tsne_sweep.csv", "umap_sweep.csv"]:
        pd.testing.assert_frame_equal(read(source / "data" / name), read(ASSETS / "aufgabe_02/data" / name))
    for name in ["subsample.png", "arcsinh.pdf", "pca_spectrum.pdf", "tsne_sweep.png", "umap_grid_1.pdf", "umap_grid_2.pdf"]:
        shutil.copyfile(source / "figures" / name, destination / name)
    for name in ["stability_eligibility.png"]:
        shutil.copyfile(ROOT / "figures" / name, destination / name)
    for name in ["task5_group_profiles_recurrence.pdf", "task5_oof_marker_profiles.pdf"]:
        shutil.copyfile(ASSETS / "aufgaben_04_05/figures" / name, destination / name)
    for name in ["network_metrics.pdf", "paired_differences.pdf", "donor_frequencies.pdf", "learned_alphas.pdf"]:
        shutil.copyfile(ROOT / "praesentation/aufgabe_06_vergleich/figures" / name, destination / f"bonus_{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruct-clusters", action="store_true")
    args = parser.parse_args()
    for directory in [OUT / "figures", OUT / "data"]: directory.mkdir(exist_ok=True)
    if args.reconstruct_clusters:
        reconstruct_clusters()
    manifest = json.loads((OUT / "data/cluster_reconstruction.json").read_text())
    for name, expected in manifest["output_sha256"].items():
        if sha(OUT / "data" / name) != expected: raise ValueError(f"Cluster export changed: {name}")
    path = figures()
    copy_supplement()
    (OUT / "data/figure_provenance.json").write_text(json.dumps({
        "exporter_sha256": sha(__file__), "task2_coordinates_sha256": sha(path),
        "task2_coordinates": str(path.relative_to(ROOT)),
        "task2_checks": "All three Procrustes disparities recomputed and matched at 1e-12",
        "task3": manifest,
        "inputs": {str(p.relative_to(ROOT)): sha(p) for p in [ASSETS / "aufgabe_02/data/provenance.json", ASSETS / "aufgaben_04_05/data/provenance.json", ASSETS / "aufgabe_06/data/provenance.json"]},
        "figures_sha256": {str(p.relative_to(OUT)): sha(p) for p in sorted((OUT / "figures").rglob("*")) if p.is_file()}
    }, indent=2) + "\n")
    print("Report figures exported.", flush=True)


if __name__ == "__main__":
    main()
