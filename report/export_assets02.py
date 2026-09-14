"""Render the three V2 main figures from checked exports, without prose footers.

No fitting, clustering, donor scoring or selection is performed here. V1 assets
and their renderers stay unchanged. The drawing conventions follow V1.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ssbi-report02-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.spatial import procrustes

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
OUT = REPORT / "figures/v02"
DATA = REPORT / "data/v02"
ASSETS = ROOT / "praesentation/report_assets"
METHODS = ["CellCNN", "SVM", "Citrus"]
MARKERS = ["CD3", "CD19", "CD56", "CD16", "CD94", "NKG2A", "NKG2C", "CD57"]
COLORS = ["#087f8c", "#6263aa", "#d98c30"]
inputs = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    inputs[str(path.relative_to(ROOT))] = sha(path)
    return path


def read(path):
    return pd.read_csv(record(path), float_precision="round_trip")


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", dpi=400, bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def dimensionality_reduction():
    coordinates = ROOT / "results/tables/report_preparation_20260913/embeddings_full.parquet"
    embeddings = pd.read_parquet(record(coordinates))
    labels = read(ROOT / "labels.csv").set_index("sample_id").clinical_group
    assert embeddings.clinical_group.eq(embeddings.sample_id.map(labels)).all()
    metadata = pd.read_parquet(record(ROOT / "nk_balanced_raw.parquet"))[["sample_id", "clinical_group"]]
    pd.testing.assert_frame_equal(embeddings[["sample_id", "clinical_group"]].reset_index(drop=True),
                                  metadata.reset_index(drop=True), check_dtype=False)
    assert len(embeddings) == 40000 and embeddings.groupby("sample_id").size().eq(2000).all()
    quality = read(ASSETS / "aufgabe_02/data/quality.csv").set_index("embedding")
    pairs = read(ASSETS / "aufgabe_02/data/pairwise.csv")
    columns = {"PCA": ["PC1", "PC2"], "t-SNE": ["tsne1", "tsne2"], "UMAP": ["umap1", "umap2"]}
    for row in pairs.itertuples():
        a, b = row.pair.split(" vs ")
        np.testing.assert_allclose(procrustes(embeddings[columns[a]], embeddings[columns[b]])[2],
                                   row.procrustes_disparity, atol=1e-12, rtol=0)
    order = np.random.default_rng(42).permutation(len(embeddings))
    colors = np.where(embeddings.clinical_group.eq("CMV+"), "#d1495b", "#2878b5")
    with plt.rc_context({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "text.color": "#23364a", "axes.labelcolor": "#23364a"}):
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.18), layout="constrained")
        subtitles = {"PCA": "PC1–PC2", "t-SNE": "perplexity 70", "UMAP": "n=5, min_dist=0, lr=0.1"}
        for ax, (name, cols) in zip(axes, columns.items()):
            points = embeddings[cols].to_numpy()
            ax.scatter(points[order, 0], points[order, 1], c=colors[order], s=1.3,
                       alpha=.5, linewidths=0, rasterized=True)
            ax.set(title=f"{name} · T={quality.loc[name, 'trustworthiness']:.3f}\n{subtitles[name]}",
                   xticks=[], yticks=[], aspect="equal")
            for spine in ax.spines.values():
                spine.set_visible(False)
        fig.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor=color,
                                   label=name, markersize=4)
                            for name, color in [("CMV−", "#2878b5"), ("CMV+", "#d1495b")]],
                   loc="outside upper center", ncol=2, frameon=False, fontsize=7.5)
        save(fig, "dimensionality_reduction")


def classification_and_subsets():
    directory = ASSETS / "aufgaben_04_05/data"
    provenance = json.loads(record(directory / "provenance.json").read_text())
    frames = {}
    for name, expected in provenance["output_sha256"].items():
        assert sha(directory / name) == expected, name
        frames[Path(name).stem] = read(directory / name)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
                         "axes.labelcolor": "#243449", "text.color": "#243449"})

    def boxes(ax, arrays, labels, colors):
        box = ax.boxplot(arrays, positions=np.arange(len(arrays)), widths=.48,
                         patch_artist=True, showfliers=False,
                         medianprops={"color": "#243449", "linewidth": 1.2})
        rng = np.random.default_rng(4531)
        for i, (v, color, patch) in enumerate(zip(arrays, colors, box["boxes"])):
            patch.set(facecolor=color, edgecolor=color, alpha=.22)
            ax.scatter(i + rng.uniform(-.13, .13, len(v)), v, color=color,
                       alpha=.65, s=12, linewidths=0)
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.grid(axis="y", alpha=.15)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.18), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.25]})
    metrics = frames["split_metrics"]
    boxes(axes[0], [metrics.loc[metrics.method.eq(m), "roc_auc"].to_numpy() for m in METHODS], METHODS, COLORS)
    axes[0].axhline(.5, color="#7d8996", ls="--", lw=.7)
    axes[0].set(title="A  Classification performance", ylabel="Test ROC-AUC", ylim=(-.04, 1.04))
    differences, paired = frames["paired_differences"], frames["paired_summary"]
    arrays, labels = [], []
    for row in paired.itertuples():
        arrays.append(differences.loc[differences["first"].eq(row.first) & differences.second.eq(row.second),
                                      "auc_difference"].to_numpy())
        labels.append(f"{row.first} − {row.second}\n{row.better} / {row.equal} / {row.worse}")
    boxes(axes[1], arrays, labels, [COLORS[0], COLORS[0], COLORS[1]])
    axes[1].axhline(0, color="#7d8996", ls="--", lw=.8)
    axes[1].set(title="B  Paired differences on the same splits", ylabel="ROC-AUC difference")
    axes[1].tick_params(axis="x", labelsize=7)
    save(fig, "classification")

    fig = plt.figure(figsize=(6.4, 3.03), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.1])
    top = grid[0].subgridspec(1, 3)
    axes = [fig.add_subplot(top[i]) for i in range(3)]
    freq = frames["cell_selection_frequencies"]
    for index, (ax, method) in enumerate(zip(axes, METHODS)):
        frame = freq.loc[freq.method.eq(method)]
        ax.scatter(frame.component_1, frame.component_2, s=1.5, color="#dedfe3", linewidths=0, rasterized=True)
        selected = frame.loc[frame.positive_frequency.gt(0)].sort_values("positive_frequency")
        ax.scatter(selected.component_1, selected.component_2, c=selected.positive_frequency,
                   vmin=0, vmax=1, cmap="viridis", s=7, linewidths=0, rasterized=True)
        ax.set(title=f"{chr(65 + index)}  {method}", aspect="equal", xticks=[], yticks=[],
               ylabel="t-SNE 2" if index == 0 else "")
        for spine in ax.spines.values():
            spine.set_visible(False)
        stats = provenance["results"][method]
        ax.set_xlabel(f"≥1 visit: {stats['selected_at_least_once']}; ≥50%: {stats['selected_at_least_half']}", fontsize=7)
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="viridis"), ax=axes,
                 label="Positive selection frequency", ticks=[0, .5, 1], shrink=.85, pad=.015, fraction=.025)
    ax = fig.add_subplot(grid[1])
    profile = frames["oof_marker_profiles"]
    values = profile.pivot(index="method", columns="marker", values="mean_z").loc[METHODS, MARKERS].to_numpy()
    limits = [profile.loc[profile.marker.isin(MARKERS), "mean_z"].abs().max(),
              frames["group_marker_profiles"].loc[frames["group_marker_profiles"].marker.isin(MARKERS), "median_z"].abs().max()]
    limit = max(1, int(np.ceil(max(limits))))
    assert limit == 5, "Keep the verified V1 marker colour scale."
    im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(MARKERS)), MARKERS)
    ax.set_yticks(range(len(METHODS)), METHODS)
    ax.tick_params(length=0)
    for i, j in np.ndindex(values.shape):
        value = 0. if abs(values[i, j]) < .05 else values[i, j]
        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7,
                color="white" if abs(values[i, j]) > .65 * limit else "#243449")
    ax.set_title("D  Selected test-cell profiles · equal donor weights", loc="left")
    fig.colorbar(im, ax=ax, label="Mean marker z-score", shrink=.95, pad=.02)
    save(fig, "subsets")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    dimensionality_reduction()
    classification_and_subsets()
    manifest = {"renderer_sha256": sha(__file__), "inputs_sha256": inputs,
                "changes": "Remove explanatory footers and reduce canvas height. Same data, ordering, scales and plot encodings as V1.",
                "figures_sha256": {str(p.relative_to(REPORT)): sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}}
    (DATA / "figure_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Exported three V2 figures from unchanged result exports.")


if __name__ == "__main__":
    main()
