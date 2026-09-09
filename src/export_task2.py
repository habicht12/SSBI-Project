"""Export Task 2 documents' assets without training any embedding models.

Run from the repository root: python -m src.export_task2
Use --update-notebook only to apply the documented partial notebook correction.
"""

import argparse
import base64
import hashlib
from importlib.metadata import version
from io import BytesIO, StringIO
from itertools import combinations
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from threadpoolctl import threadpool_limits

from src.task2_metrics import neighbor_indices, neighborhood_overlap

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/02_dimensionality_reduction_clean.ipynb"
OUT = ROOT / "praesentation/aufgabe_02"
INK, TEAL, RUST = "#172B43", "#087F8C", "#AF4D35"


def find_cell(nb, fragment):
    matches = [c for c in nb["cells"] if fragment in "".join(c.get("source", []))]
    if len(matches) != 1:
        raise ValueError(f"Expected one source cell containing {fragment!r}; found {len(matches)}")
    return matches[0]


def read_table(cell, index_col=None):
    html = next(o["data"]["text/html"] for o in cell.get("outputs", [])
                if "text/html" in o.get("data", {}))
    table = pd.read_html(StringIO("".join(html)), index_col=index_col)[0]
    if isinstance(table.columns, pd.MultiIndex):
        table.index.name = table.columns.names[-1]
        table.columns = table.columns.get_level_values(0)
    return table


def table_output(frame):
    return {"data": {"text/html": [frame.to_html()],
                     "text/plain": [frame.to_string()]},
            "metadata": {}, "output_type": "display_data"}


def figure_output(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    return {"data": {"image/png": base64.b64encode(buffer.getvalue()).decode()},
            "metadata": {}, "output_type": "display_data"}


def set_source(cell, source):
    cell["source"] = source.splitlines(keepends=True)


def panel_figure(panels, labels, filename, columns=2):
    """Retain original plotted pixels; replace tiny raster titles by vector labels."""
    rows = len(panels) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(1.6*columns, 1.6*rows), squeeze=False)
    for ax, panel, label in zip(axes.flat, panels, labels):
        # All source panels have their title above y=35; plotted axes begin at y=38.
        ax.imshow(panel.crop((0, 35, panel.width, panel.height-3)))
        ax.set_axis_off()
        ax.set_title(label, fontsize=12 if columns==2 else 9, pad=3)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=.92, wspace=.05, hspace=.30)
    fig.savefig(OUT / "figures" / filename)
    plt.close(fig)


def export_assets(update_notebook=False):
    nb = json.loads(NOTEBOOK.read_text())
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": INK,
                         "text.color": INK, "axes.titlecolor": INK,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})

    image_sources = [
        ("comparison = pd.concat", ["subsample.png"]),
        ("tsne_sweep_labels =", ["tsne_sweep.png"]),
        ("umap_sweep_labels =", ["umap_sweep.png"]),
        ("diagnostic_markers_for_overlay =", ["umap_markers.png", "tsne_markers.png"]),
        ("SHEPARD_N_SAMPLE =", ["shepard.png"]),
    ]
    provenance_images = {}
    for fragment, names in image_sources:
        cell = find_cell(nb, fragment)
        images = [o["data"]["image/png"] for o in cell["outputs"]
                  if "image/png" in o.get("data", {})]
        assert len(images) == len(names), fragment
        for name, encoded in zip(names, images):
            (OUT / "figures" / name).write_bytes(base64.b64decode("".join(encoded)))
            provenance_images[name] = {"source_anchor": fragment,
                                      "cell_ordinal": nb["cells"].index(cell) + 1}

    # Re-arrange saved sweep panels only; the embedded cell coordinates are unchanged.
    im = Image.open(OUT / "figures/tsne_sweep.png").convert("RGB")
    width = im.width / 6
    panels = [im.crop((round(i * width), 0, round((i + 1) * width), im.height))
              for i in (0, 1, 3, 5)]
    panel_figure(panels, [f"Perplexity {p}" for p in (5,30,60,100)], "tsne_selected.pdf")
    im = Image.open(OUT / "figures/umap_sweep.png").convert("RGB")
    tile = im.width / 4
    height = im.height / 4
    umap_panels = []
    for j, (row, col) in enumerate(((0, 0), (0, 3), (3, 0), (3, 3))):
        panel = im.crop((round(col * tile), round(row * height),
                         round((col + 1) * tile), round((row + 1) * height)))
        umap_panels.append(panel)
    panel_figure(umap_panels, ["n=5, d=0", "n=5, d=0.5", "n=50, d=0", "n=50, d=0.5"], "umap_selected.pdf")

    raw = pd.read_parquet(ROOT / "nk_balanced_raw.parquet")
    emb = pd.read_parquet(ROOT / "embeddings_full.parquet")
    reference = np.load(ROOT / "X_pca_full.npy")
    assert reference.shape == (40000, 29)
    assert emb[["sample_id", "clinical_group"]].equals(
        raw[["sample_id", "clinical_group"]].reset_index(drop=True))
    assert raw.groupby("sample_id").size().eq(2000).all()
    assert raw["sample_id"].nunique() == 20
    coords = {name: emb[cols].to_numpy() for name, cols in [
        ("PCA", ["PC1", "PC2"]), ("t-SNE", ["tsne1", "tsne2"]),
        ("UMAP", ["umap1", "umap2"]) ]}
    assert np.array_equal(reference[:, :2], coords["PCA"])
    assert all(np.isfinite(x).all() for x in [reference, *coords.values()])

    print("Computing four neighbour index matrices, single-threaded, 64 MiB working memory.", flush=True)
    indices = {}
    for name, values in {"reference": reference, **coords}.items():
        indices[name] = neighbor_indices(values)
        assert indices[name].shape == (40000, 15)
        assert all(i not in row for i, row in enumerate(indices[name]))
        print(f"  {name}: done", flush=True)

    quality_cell = find_cell(nb, "quality_rows = []")
    quality = read_table(quality_cell, index_col=0)
    for name in coords:
        quality.loc[name, "knn_preservation"] = neighborhood_overlap(indices["reference"], indices[name])
    sample = np.random.default_rng(42).choice(len(reference), 2000, replace=False)
    d_ref = pdist(reference[sample])
    for name, values in coords.items():
        r = pearsonr(d_ref, pdist(values[sample])).statistic
        assert abs(quality.loc[name, "distance_correlation"] - r) < 0.00000051
        quality.loc[name, "distance_correlation"] = r
    pair_rows = []
    for a, b in combinations(coords, 2):
        pair_rows.append({"pair": f"{a} vs {b}",
                          "knn_preservation": neighborhood_overlap(indices[a], indices[b]),
                          "procrustes_disparity": procrustes(coords[a], coords[b])[2]})
    pairwise = pd.DataFrame(pair_rows)
    old_pairs = read_table(find_cell(nb, "pair_rows = []"), index_col=0)
    assert np.allclose(old_pairs["procrustes_disparity"], pairwise["procrustes_disparity"], atol=5.1e-7)
    quality.to_csv(OUT / "data/quality.csv", float_format="%.9f")
    pairwise.to_csv(OUT / "data/pairwise.csv", index=False, float_format="%.9f")

    tsne_cell = find_cell(nb, "tsne_param_grid =")
    umap_cell = find_cell(nb, "umap_param_grid =")
    tsne = read_table(tsne_cell, index_col=0).drop(columns="knn_preservation", errors="ignore")
    umap = read_table(umap_cell, index_col=0).drop(columns="knn_preservation", errors="ignore")
    assert int(tsne.sort_values("trustworthiness", ascending=False).iloc[0]["perplexity"]) == 60
    selected_umap = umap.sort_values("trustworthiness", ascending=False).iloc[0]
    assert (selected_umap["n_neighbors"], selected_umap["min_dist"]) == (5, 0)
    for part, neighbors in enumerate(((5,15),(30,50))):
        panels, titles = [], []
        for row, nn in enumerate(neighbors, start=part*2):
            for col, md in enumerate((0,.1,.3,.5)):
                panels.append(im.crop((round(col*tile), round(row*height), round((col+1)*tile), round((row+1)*height))))
                tw = float(umap.loc[(umap.n_neighbors==nn) & (umap.min_dist==md), "trustworthiness"].iloc[0])
                titles.append(f"n={nn}, d={md:g}\nT={tw:.3f}")
        panel_figure(panels, titles, f"umap_grid_{part+1}.pdf", columns=4)
    tsne.to_csv(OUT / "data/tsne_sweep.csv", index=False)
    umap.to_csv(OUT / "data/umap_sweep.csv", index=False)

    def score(v, best, worst, places=4):
        s = f"{v:.{places}f}"
        return (r"\best{" + s + "}") if v == best else ((r"\worst{" + s + "}") if v == worst else s)
    columns = ["trustworthiness", "knn_preservation", "distance_correlation", "silhouette_cmv"]
    lines = [r"\begin{tabular}{lrrrr}\toprule", r"Embedding & $T(15)\uparrow$ & $R(15)\uparrow$ & $r_d\uparrow$ & $S_{\rm CMV}\uparrow$\\\midrule"]
    for name in coords:
        lines.append(name + " & " + " & ".join(score(quality.loc[name,c], quality[c].max(), quality[c].min()) for c in columns) + r"\\")
    (OUT / "data/quality.tex").write_text("\n".join(lines + [r"\bottomrule\end{tabular}"]) + "\n")
    lines = [r"\begin{tabular}{lrr}\toprule", r"Pair & $R(15)\uparrow$ & $M^2\downarrow$\\\midrule"]
    for _, row in pairwise.iterrows():
        lines.append(row["pair"] + " & " + score(row["knn_preservation"], pairwise.knn_preservation.max(), pairwise.knn_preservation.min(), 5)
                     + " & " + score(row["procrustes_disparity"], pairwise.procrustes_disparity.min(), pairwise.procrustes_disparity.max(), 5) + r"\\")
    (OUT / "data/pairwise.tex").write_text("\n".join(lines + [r"\bottomrule\end{tabular}"]) + "\n")
    lines = [r"\begin{tabular}{rrr}\toprule", r"$n_{\rm neighbors}$ & min\_dist & $T(15)$\\\midrule"]
    for _, row in umap.sort_values("trustworthiness", ascending=False).head(3).iterrows():
        lines.append(f"{int(row['n_neighbors'])} & {row['min_dist']:.1f} & {row['trustworthiness']:.6f}" + r"\\")
    (OUT / "data/umap_top.tex").write_text("\n".join(lines + [r"\bottomrule\end{tabular}"]) + "\n")
    lines = [r"\begin{tabular}{rr}\toprule", r"Perplexity & $T(15)$\\\midrule"]
    for _, row in tsne.sort_values("perplexity").iterrows():
        lines.append(f"{int(row['perplexity'])} & {row['trustworthiness']:.6f}" + r"\\")
    (OUT / "data/tsne_sweep.tex").write_text("\n".join(lines + [r"\bottomrule\end{tabular}"]) + "\n")

    channels = pd.read_parquet(ROOT / "channel_table.parquet")
    markers = channels.loc[channels.analysis_marker, "marker_name_pns"].tolist()
    transformed = np.arcsinh(raw[markers].to_numpy() / 5)
    scaled = (transformed - transformed.mean(axis=0)) / transformed.std(axis=0, ddof=1)
    variance = np.linalg.eigvalsh(np.cov(scaled, rowvar=False))[::-1]
    ratio = variance / variance.sum()
    cumulative = np.cumsum(ratio)
    assert np.searchsorted(cumulative, .90) + 1 == 29
    (OUT / "data/numbers.tex").write_text(
        f"\\newcommand{{\\PCOne}}{{{100*ratio[0]:.2f}}}\n"
        f"\\newcommand{{\\PCTwo}}{{{100*ratio[1]:.2f}}}\n"
        f"\\newcommand{{\\PCTogether}}{{{100*cumulative[1]:.2f}}}\n"
        f"\\newcommand{{\\PCSelected}}{{{100*cumulative[28]:.2f}}}\n")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.linspace(-30, 100, 600)
    ax.plot(x, np.arcsinh(x / 5), color=TEAL, lw=2, label=r"$\operatorname{asinh}(x/5)$")
    ax.plot(x, x / 5, color=RUST, ls="--", label=r"linear approximation $x/5$")
    ax.axhline(0, color=".8", lw=.7); ax.axvline(0, color=".8", lw=.7)
    ax.set(xlabel="Raw marker intensity (illustrative units)", ylabel="Transformed intensity", ylim=(-3.5, 4.5))
    ax.legend(fontsize=9, loc="lower right")
    fig.savefig(OUT / "figures/arcsinh.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.plot(np.arange(1,31), cumulative[:30]*100, "o-", color=TEAL, ms=3)
    ax.axhline(90, color=RUST, ls="--", lw=1)
    ax.scatter([29], [cumulative[28]*100], color=RUST, zorder=4)
    ax.annotate(f"29 PCs: {100*cumulative[28]:.2f}%", (29,cumulative[28]*100), xytext=(14,65), arrowprops={"arrowstyle":"->","color":RUST})
    ax.set(xlabel="Number of retained PCs", ylabel="Cumulative explained variance (%)", ylim=(0,100), xlim=(0,31))
    fig.savefig(OUT / "figures/pca_spectrum.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    donors = sorted(emb.sample_id.unique())
    for i, donor in enumerate(donors):
        mask = emb.sample_id.eq(donor)
        ax.scatter(coords["PCA"][mask,0], coords["PCA"][mask,1], s=1, alpha=.5, color=plt.get_cmap("tab20")(i), rasterized=True, label=donor)
    ax.set(xlabel=f"PC1 ({100*ratio[0]:.2f}%)", ylabel=f"PC2 ({100*ratio[1]:.2f}%)", xticks=[], yticks=[])
    ax.legend(ncol=2, bbox_to_anchor=(1,1), loc="upper left", fontsize=7, markerscale=4, handletextpad=.2, columnspacing=.5, frameon=False)
    fig.savefig(OUT / "figures/pca_donor.pdf", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(10,2.7))
    groups = sorted(emb.clinical_group.unique())
    for ax, (name, values) in zip(axes, coords.items()):
        for group in groups:
            mask = emb.clinical_group.eq(group)
            color = RUST if group == "CMV+" else "#2878b5"
            ax.scatter(values[mask,0], values[mask,1], s=1, alpha=.5, color=color, label=group, rasterized=True)
        ax.set(title=name, xticks=[], yticks=[])
    axes[-1].legend(fontsize=9, markerscale=4, frameon=False)
    fig.savefig(OUT / "figures/embeddings_cmv.pdf", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4,2.7))
    d = np.linspace(0,1,101)
    ax.plot(d,d,color=TEAL,lw=2,label="equal distances")
    ax.plot(d,.6*d,color=RUST,ls="--",label="uniform rescaling")
    ax.set(xlabel="Reference distance (schematic)", ylabel="Embedding distance", xlim=(0,1), ylim=(0,1))
    ax.legend(fontsize=8)
    fig.savefig(OUT / "figures/shepard_ideal.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(8,3.4))
    for ax, metric, diagonal, title, higher in [
        (axes[0],"knn_preservation",1,"Shared neighbours, k=15",True),
        (axes[1],"procrustes_disparity",0,"Procrustes disparity",False)]:
        matrix = np.eye(3) * diagonal
        for row,(a,b) in zip(pair_rows,combinations(range(3),2)):
            matrix[a,b] = matrix[b,a] = row[metric]
        ax.imshow(matrix,cmap="viridis" if higher else "viridis_r",vmin=0,vmax=1)
        ax.set(xticks=range(3),yticks=range(3),xticklabels=coords.keys(),yticklabels=coords.keys(),title=title)
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f"{matrix[i,j]:.3f}",ha="center",va="center",color="black" if (matrix[i,j]>.55 if higher else matrix[i,j]<.45) else "white",fontsize=10)
    fig.tight_layout()
    pair_image = figure_output(fig)
    fig.savefig(OUT / "figures/pairwise.pdf"); plt.close(fig)

    if update_notebook:
        function_cell = find_cell(nb, "def knn_preservation") if any("def knn_preservation" in "".join(c.get("source",[])) for c in nb["cells"]) else find_cell(nb, "from src.task2_metrics import knn_preservation")
        source = "".join(function_cell["source"])
        plot_source = source[source.index("def plot_param_sweep_grid"):]
        set_source(function_cell, "# The helper requests k neighbours with X=None; sklearn already excludes self.\nimport sys\nif str(PROJECT_ROOT) not in sys.path:\n    sys.path.insert(0, str(PROJECT_ROOT))\nfrom src.task2_metrics import knn_preservation\n\n\n" + plot_source)
        function_cell["execution_count"] = None
        load_cell = find_cell(nb, "def find_project_root")
        set_source(load_cell, "".join(load_cell["source"]).replace('(candidate / "NK_cell_dataset").exists()', '(candidate / "nk_balanced_raw.parquet").exists()'))
        set_source(quality_cell, "".join(quality_cell["source"]).replace("reference_space = X_pca_full", "reference_space = X_pca  # the 29 selected PCs, matching the saved outputs"))
        quality_cell["outputs"] = [table_output(quality)]
        quality_cell["execution_count"] = None
        pair_cell = find_cell(nb, "pair_rows = []")
        pair_cell["outputs"] = [table_output(pairwise), pair_image]
        pair_cell["execution_count"] = None
        for cell, table in [(tsne_cell,tsne),(umap_cell,umap)]:
            # Old kNN outputs are invalid; embeddings and Trustworthiness were not rerun.
            cell["outputs"] = [table_output(table)]
            cell["execution_count"] = None
        context = find_cell(nb, "## 10. Quantitative comparison")
        s = "".join(context["source"]).replace("(`X_pca_full`)", "(`X_pca`, 29 selected PCs)")
        s = s.replace("*Distance correlation*", "*Pearson correlation of pairwise distances*")
        s += ("\n\n**Partial correction (2026-09-09):** kNN overlap now uses exactly 15 neighbours excluding self. "
              "Final kNN values and pairwise Procrustes were recalculated from the committed 40,000-cell exports; "
              "Pearson distance correlations were checked on the fixed 2,000-cell evaluation sample. "
              "The 29-PC reference matches the stored current distance correlations. "
              "Trustworthiness and CMV silhouette remain saved notebook results, not newly verified full-data runs. "
              "Sweep embeddings were not retrained: invalid saved sweep kNN columns were removed; "
              "a future full execution will calculate them with the corrected function. "
              "Parameter selection still uses Trustworthiness alone. See `praesentation/aufgabe_02/README.md`.\n") if "**Partial correction (2026-09-09):**" not in s else ""
        set_source(context,s)
        corrections = {
            "**Cofactor choice: 5.**": "**Cofactor choice: 5.** The histogram grid documents a qualitative cofactor choice. The arcsinh transform is approximately linear near zero and logarithmic for large magnitudes; changing a positive cofactor preserves the sign and ordering of intensities. The grid is not a formal demonstration that c=5 is optimal for every marker.\n",
            "If points fall close to the diagonal": "Points near the diagonal support retention of this particular GMM-defined NKG2C+/CD57+ fraction. The empty table indicates that no donor has fewer than ten flagged cells in the subsample. This does not establish representativeness for every rare population or validate the approximate positivity gates.\n",
            "## 4. PCA": "## 4. PCA\n\n### Parameter: number of components\n\nWe investigate how many components to retain using the explained-variance spectrum. Retained dimensionality is distinct from the displayed axis pair. Other choices include preprocessing, solver and whitening; these are held fixed here. We first fit 30 components, then retain 29 to exceed 90% cumulative explained variance.\n",
            "**Shepard diagrams**": "**Shepard diagrams** compare reference and embedding distances for the same sampled cell pairs. Exact distance preservation gives the identity line; uniform positive rescaling gives another straight line with Pearson correlation 1. The diagonal is therefore not necessary for perfect correlation.\n",
        }
        for fragment, replacement in corrections.items():
            anchor = fragment if any(fragment in "".join(c.get("source", [])) for c in nb["cells"]) else replacement.strip()
            set_source(find_cell(nb, anchor), replacement)
        summary = find_cell(nb,"## 11. Summary")
        set_source(summary, "## 11. Summary\n\n"
                   "- **Data:** 40,000 gated NK cells (2,000 per donor; 20 donors), 37 retained markers; arcsinh(x/5), then pooled marker-wise z-scoring. This is exploratory preprocessing, not a cross-validation pipeline.\n"
                   "- **Sampling check:** the saved full-versus-subsample scatter shows close agreement for the GMM-defined NKG2C+/CD57+ fraction. No donor has fewer than 10 flagged cells. This does not establish representativeness for every rare population or validate the biological gate.\n"
                   f"- **PCA:** 29 PCs explain {100*cumulative[28]:.2f}% of variance; the first two explain only {100*cumulative[1]:.2f}%.\n"
                   "- **Parameter selection:** t-SNE perplexity 60 has the highest saved Trustworthiness (0.926046), only slightly above 50 (0.925973). UMAP selects n_neighbors=5 and min_dist=0.0. The fixed sweep has 5,000 cells and 29 input PCs.\n"
                   "- **Quality:** saved Trustworthiness favours t-SNE; Pearson distance correlation favours PCA. CMV silhouettes are small (0.0016 to 0.0178), so none is evidence of a strong global class separation or donor-level predictive performance.\n"
                   "- **Pairwise results:** corrected kNN overlap is 0.005850 for PCA/t-SNE, 0.004095 for PCA/UMAP, and 0.125318 for t-SNE/UMAP. Procrustes disparity is 0.451891, 0.733897, and 0.723692, respectively. Local-neighbour similarity and globally aligned shape therefore give different rankings.\n"
                   "- **Biology:** marker overlays are exploratory maps of expression. NKG2C/CD57 co-expression is a phenotype associated with memory-like NK cells, not a per-cell infection label or functional validation.\n")
        NOTEBOOK.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n")

    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in [
        "embeddings_full.parquet", "X_pca_full.npy", "nk_balanced_raw.parquet", "channel_table.parquet"]}
    provenance = {"source_commit":"252415a64ec23cc66772a86f636ba01f4d0f3e4b",
                  "notebook_sha256":hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
                  "input_sha256":hashes,"n_cells":40000,"n_donors":20,"cells_per_donor":2000,
                  "reference_pcs":29,"k":15,"seed":42,"distance_sample_cells":2000,
                  "software":{"python":platform.python_version(), **{name:version(name) for name in ["numpy","pandas","scipy","scikit-learn","matplotlib","pillow"]}},
                  "saved_image_sources":provenance_images,
                  "recalculated":["kNN quality", "pairwise kNN", "Procrustes", "Pearson distance correlation", "PCA variance spectrum"],
                  "retained_without_retraining":["t-SNE/UMAP coordinates and sweep plots", "Trustworthiness", "CMV silhouette", "GMM subsampling plot"],
                  "pca_variance":{"PC1":float(ratio[0]),"PC2":float(ratio[1]),"PC29_cumulative":float(cumulative[28])}}
    (OUT/"data/provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(quality.to_string(),flush=True)
    print(pairwise.to_string(index=False),flush=True)
    print(f"Assets exported to {OUT}; no embedding model was trained.",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-notebook", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        export_assets(args.update_notebook)
