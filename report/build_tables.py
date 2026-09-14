"""Generate the supplement's numeric tables directly from frozen CSVs."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ssbi-report-mpl")
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/tables"
A = ROOT / "praesentation/report_assets"
OUT.mkdir(exist_ok=True)
read = lambda p: pd.read_csv(p, float_precision="round_trip")


def table(name, frame, long=False, caption=None, label=None):
    text = frame.to_latex(index=False, escape=True, longtable=long, na_rep="—",
                          float_format="%.4f", caption=caption, label=label,
                          column_format="l" + "r" * (len(frame.columns)-1))
    (OUT / f"{name}.tex").write_text(text)


labels = read(ROOT / "labels.csv").set_index("sample_id")
alive = read(A / "aufgaben_04_05/data/citrus_input_checks.csv").set_index("donor_id")
nk = pd.read_parquet(ROOT / "nk_full_raw.parquet").groupby("sample_id").size()
freq = read(A / "aufgaben_04_05/data/cell_selection_frequencies.csv")
visits = freq.groupby("sample_id").n_test_models.first()
donors = pd.DataFrame({"Donor": alive.index, "CMV": labels.loc[alive.index, "clinical_group"].values,
                       "Alive": alive.n_cells.values, "NK": nk.loc[alive.index].values,
                       "NK / alive (pct)": 100 * nk.loc[alive.index].values / alive.n_cells.values,
                       "Test visits (30)": visits.loc[alive.index].values})
donors["CMV"] = donors.CMV.replace({"CMV−": "CMV-"})
table("donors", donors)
markers = json.loads((A / "aufgabe_02/data/provenance.json").read_text())["markers"]
chunks = [markers[i:i+10] for i in range(0, len(markers), 10)]
marker_table = pd.DataFrame({f"Panel {i+1}": column+[""]*(10-len(column)) for i,column in enumerate(chunks)})
table("markers", marker_table)
q = read(A / "aufgabe_02/data/quality.csv").drop(columns="silhouette_cmv")
q.columns = ["Embedding", "Trustworthiness", "kNN overlap", "Distance r"]
table("quality", q)
pairs = read(A / "aufgabe_02/data/pairwise.csv")
pairs.columns = ["Pair", "kNN overlap", "Procrustes"]
table("pairs", pairs)
t = read(A / "aufgabe_02/data/tsne_sweep.csv")
t.columns = ["Perplexity", "Trustworthiness", "kNN overlap"]
table("tsne", t)
u = read(A / "aufgabe_02/data/umap_sweep.csv")
u = u[["learning_rate", "n_neighbors", "min_dist", "trustworthiness", "knn_preservation"]]
u.columns = ["Learning rate", "Neighbours", "Min. distance", "Trustworthiness", "kNN overlap"]
table("umap", u.sort_values(list(u.columns[:3])), long=True,
      caption="All 64 UMAP configurations on the fixed 5,000-cell sweep sample.", label="tab:umapgrid")
lr = read(A / "aufgabe_02/data/umap_learning_rate_comparison.csv")
lr = lr.groupby("learning_rate", as_index=False).trustworthiness.mean()
lr.columns = ["Learning rate", "Mean trustworthiness (two fixed settings)"]
table("learning_rate", lr)
g = read(ROOT / "data/grid.csv")
g["method"] = g.method.replace({f"Hierarchical ({x})": x.capitalize() for x in ["single","average","complete","ward"]})
g = g[["id", "method", "k", "neighbors", "resolution", "clusters", "score", "stability"]]
for col in ["k", "neighbors"]:g[col] = g[col].map(lambda v: str(int(v)) if pd.notna(v) else "--")
g["resolution"] = g.resolution.map(lambda v: f"{v:.1f}" if pd.notna(v) else "--")
g.columns = ["ID", "Method", "k", "Neighbours", "Resolution", "Clusters", "DB", "Stability"]
table("clustering_grid", g, long=True, caption="The original 44 clustering configurations. Scores and stability are stored at three-decimal precision.", label="tab:clustergrid")
c = read(ROOT / "data/selected_cluster_stability.csv")
c = c.loc[c.method.isin(["k-means", "Hierarchical (ward)", "Leiden"])].copy()
c["method"] = c.method.replace({"Hierarchical (ward)": "Ward"})
c = c[["method", "cluster", "original_cells", "median_jaccard", "q25", "q75", "evaluated_repeats"]]
c.columns = ["Method", "Cluster", "Cells", "Median J", "Q1", "Q3", "Repeats"]
table("cluster_stability", c, long=True, caption="Cluster-wise stability for the three illustrated partitions.", label="tab:clusterstability")
metrics = read(A / "aufgaben_04_05/data/metric_summary.csv")
for statistic in ["mean", "median"]:
    s = metrics.pivot(index="method", columns="metric", values=statistic).loc[["CellCNN", "SVM", "Citrus"]]
    s = s[["roc_auc", "average_precision", "balanced_accuracy"]].reset_index()
    s.columns = ["Method", "AUC", "AP", "BA"]
    table(f"classification_{statistic}", s)
rows=[]
for method in ["CellCNN", "SVM", "Citrus"]:
    line={"Method":method}
    for col,name in [("roc_auc","AUC"),("average_precision","AP"),("balanced_accuracy","BA")]:
        r=metrics.loc[metrics.method.eq(method)&metrics.metric.eq(col)].iloc[0]
        line[name]=f"{r['median']:.3f} [{r.q1:.3f}, {r.q3:.3f}]"
    rows.append(line)
table("classification_iqr",pd.DataFrame(rows))
paired = read(A / "aufgaben_04_05/data/paired_summary.csv")
paired["Comparison"] = paired["first"] + " - " + paired["second"]
paired = paired[["Comparison", "mean", "median", "q1", "q3", "better", "equal", "worse"]]
paired.columns = ["Comparison", "Mean", "Median", "Q1", "Q3", "Better", "Equal", "Worse"]
table("paired", paired)
extra = read(ROOT / "results/tables/benchmark_main_20260912/results/tables/task4_cellcnn_svm_100_split_metrics.csv")
extra["method"] = extra.method_label.replace({"Lineare Single-Cell-SVM": "SVM"})
extra_rows=[]
for method,group in extra.groupby("method",sort=False):
    row={"Method (100 splits)":method}
    for col,label in [("roc_auc","AUC"),("average_precision","AP"),("balanced_accuracy","BA")]:
        v=group[col]
        row[label]=f"{v.median():.3f} [{v.quantile(.25):.3f}, {v.quantile(.75):.3f}]"
    extra_rows.append(row)
table("classification_100",pd.DataFrame(extra_rows))
(OUT / "clustering_winners.tex").write_text((ROOT / "data/winner_table.tex").read_text())
b = read(A / "aufgabe_06/data/summary.csv")
names={"cellcnn":"CellCNN", "quadratic_top1":"Quadratic Top-1 pct", "quadratic_soft":"Quadratic Soft-alpha", "prototype_soft":"Mahalanobis Soft-alpha"}
b["Model"] = b.model.map(names)
for suffix,columns in [("network",["network_auc_mean","network_auc_median","average_precision_mean","balanced_accuracy_mean"]),
                        ("frequency",["frequency_auc_mean","frequency_auc_median","frequency_effect_mean"] )]:
    f=b[["Model"]+columns].copy()
    f.columns=["Model"]+(["Mean AUC","Median AUC","Mean AP","Mean BA"] if suffix=="network" else ["Mean AUC","Median AUC","Mean effect (pp)"])
    if suffix=="frequency":f.iloc[:,-1]*=100
    table(f"bonus_{suffix}",f)
d=read(A / "aufgabe_06/data/paired_differences.csv")
records=[]
for model,group in d.groupby("model",sort=False):
    v=group.network_auc_delta
    records.append({"Model minus baseline":names[model],"Mean delta":v.mean(),"Median delta":v.median(),"Better":int((v>0).sum()),"Equal":int((v==0).sum()),"Worse":int((v<0).sum())})
table("bonus_paired",pd.DataFrame(records))
groups=read(A / "aufgaben_04_05/data/groups.csv")
groups=groups.loc[groups.retained,["method","group_id","n_centroids","occurrences","positive_splits","negative_splits"]]
groups.columns=["Method","Group","Centroids","Splits","Positive","Negative"]
table("groups",groups)
summary=json.loads((A / "aufgaben_04_05/data/provenance.json").read_text())["results"]
cell_summary=pd.DataFrame([{"Method":m,"Any visit":s["selected_at_least_once"],"At least half":s["selected_at_least_half"],"Every visit":s["selected_every_test_visit"],"Nonempty visits":s["nonempty_visits"],"Donors":s["contributing_donors"]} for m,s in summary.items()])
table("cell_selection",cell_summary)
(ROOT / "report/data/table_provenance.json").write_text(json.dumps({
    "generator_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "table_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.tex'))}
},indent=2)+'\n')
print(f"Generated {len(list(OUT.glob('*.tex')))} numeric tables.")
