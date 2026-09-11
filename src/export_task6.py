"""Render English Task 6 slide assets from the frozen 100-split benchmark.

Run from the repository root: python -m src.export_task6
No notebooks, model checkpoints, FCS files or GPU are needed.
Plot geometry and aggregation follow GrHa-learnable-pooling at 1f00db2.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "praesentation/aufgabe_06_vergleich"
MODELS = {
    "cellcnn": ("CellCNN · Top-1%", "#087F8C"),
    "quadratic_top1": ("Quadratic · Top-1%", "#4956A3"),
    "quadratic_soft": ("Quadratic · Soft-α", "#DD8D29"),
    "prototype_soft": ("Prototype/ReLU · Soft-α", "#AF4D35"),
}
SHORT_LABELS = ["CellCNN", "Quadratic\nTop-1%", "Quadratic\nSoft-α", "Prototype\nSoft-α"]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def common_frequency_splits(metrics):
    return set.intersection(*(set(g.loc[g.phenotype_available, "split_id"])
                              for _, g in metrics.groupby("model")))

def frequency_values(metrics, model, column, common):
    values = metrics.loc[metrics.model.eq(model) & metrics.split_id.isin(common), column]
    return values * (100 if column == "frequency_effect" else 1)

def donor_frequencies(frequencies):
    """Average visits within each donor before displaying one point per donor."""
    if frequencies.groupby("donor_id").y_true.nunique().gt(1).any():
        raise ValueError("Inconsistent donor labels.")
    return frequencies.groupby(["donor_id", "y_true"], as_index=False).agg(
        frequency=("frequency", "mean"), valid_test_visits=("frequency", "count"))

def load_data(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "input_provenance.json").read_text())
    for name, expected in manifest["input_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Benchmark input changed: {name}")
    data = {Path(name).stem: pd.read_csv(directory / name, float_precision="round_trip")
            for name in manifest["input_sha256"]}
    metrics = data["split_metrics"]
    assert set(metrics.model) == set(MODELS)
    assert not metrics.duplicated(["model", "split_id"]).any()
    for _, group in metrics.groupby("model"):
        assert set(group.split_id) == set(range(100))
    common = common_frequency_splits(metrics)
    assert common == set(metrics.loc[metrics.common_frequency_split, "split_id"])
    summary = data["summary"].set_index("model")
    for model in MODELS:
        group = metrics.loc[metrics.model.eq(model)]
        assert summary.loc[model, "frequency_splits_common"] == len(common)
        for column in ["network_auc", "average_precision", "balanced_accuracy", "frequency_auc", "frequency_effect"]:
            values = group.loc[group.split_id.isin(common), column] if column.startswith("frequency") else group[column]
            for stat, value in [("mean", values.mean()), ("median", values.median())]:
                np.testing.assert_allclose(summary.loc[model, f"{column}_{stat}"], value, atol=1e-14, rtol=0)
    for (model, kind), curve in data["mean_rocs"].groupby(["model", "kind"]):
        assert len(curve) == 501 and curve.fpr.is_monotonic_increasing
        assert curve.mean_tpr.between(0, 1).all()
        expected = summary.loc[model, f"{kind if kind == 'frequency' else 'network'}_auc_mean"]
        np.testing.assert_allclose(curve.mean_split_auc, expected, atol=1e-14, rtol=0)
        assert curve.n_splits.eq(100 if kind == "network" else len(common)).all()
    freq = data["frequencies"]
    assert not freq.duplicated(["model", "split_id", "donor_id"]).any()
    assert freq.n_cells.eq(20000).all()
    assert freq.groupby(["model", "split_id"]).size().eq(6).all()
    assert freq.groupby(["model", "split_id"]).y_true.sum().eq(2).all()
    donors = freq[["donor_id", "y_true"]].drop_duplicates()
    assert len(donors) == 20 and donors.y_true.sum() == 9
    available = freq.frequency.notna()
    np.testing.assert_allclose(freq.loc[available, "frequency"],
                               freq.loc[available, "n_selected_cells"] / freq.loc[available, "n_cells"], atol=1e-14, rtol=0)
    wide = metrics.pivot(index="split_id", columns="model", values="network_auc")
    for model, group in data["paired_differences"].groupby("model"):
        expected = wide[model] - wide.cellcnn
        np.testing.assert_allclose(group.set_index("split_id").network_auc_delta.sort_index(), expected.sort_index(), atol=1e-14)
    return data, common, manifest

def boxes(ax, values, labels, colors, *, zero=False, limits=None):
    """Keep the original quartiles, points and deterministic jitter (seed 610)."""
    arrays = [np.asarray(v, dtype=float)[np.isfinite(v)] for v in values]
    positions = np.arange(len(arrays))
    result = ax.boxplot(arrays, positions=positions, widths=.48, patch_artist=True, showfliers=False)
    rng = np.random.default_rng(610)
    for patch, array, position, color in zip(result["boxes"], arrays, positions, colors):
        patch.set(facecolor=color, alpha=.22, edgecolor=color)
        ax.scatter(position + rng.uniform(-.12, .12, len(array)), array, s=12, color=color, alpha=.5, linewidths=0)
    ax.set_xticks(positions, labels)
    if zero:
        ax.axhline(0, color="#526579", linestyle="--", lw=.8)
    if limits:
        ax.set_ylim(*limits)
    ax.grid(axis="y", alpha=.15)

def save(fig, directory, name):
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)

def plot_assets(data, common, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#172B43", "text.color": "#172B43", "axes.titleweight": "bold",
        "pdf.fonttype": 42, "ps.fonttype": 42})
    metrics = data["split_metrics"]; colors = [value[1] for value in MODELS.values()]
    for kind in ["network", "frequency"]:
        fig, ax = plt.subplots(figsize=(7.2, 3.9), layout="constrained")
        for model, (label, color) in MODELS.items():
            curve = data["mean_rocs"].loc[data["mean_rocs"].model.eq(model) & data["mean_rocs"].kind.eq(kind)].sort_values("fpr")
            ax.plot(np.r_[0, curve.fpr], np.r_[0, curve.mean_tpr], color=color, lw=2,
                linestyle="--" if model == "quadratic_soft" else "-",
                label=f"{label}: mean split AUC {curve.mean_split_auc.iloc[0]:.4f} (n={int(curve.n_splits.iloc[0])})")
        ax.plot([0, 1], [0, 1], "--", color="#a6afb9", lw=.8)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate",
               title="Network ROC · 100 donor splits" if kind == "network" else f"Half-max frequency ROC · {len(common)} common splits")
        ax.legend(fontsize=10, loc="lower right"); save(fig, directory, f"{kind}_roc")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    for ax, column, title in zip(axes, ["network_auc", "average_precision", "balanced_accuracy"],
            ["ROC-AUC", "Average precision", "Balanced accuracy (threshold 0.5)"]):
        boxes(ax, [metrics.loc[metrics.model.eq(k), column] for k in MODELS], SHORT_LABELS, colors, limits=(-.03, 1.03))
        ax.set_title(title)
    fig.supxlabel("100 overlapping donor splits. Boxes: Q1–Q3; line: median; dots: splits. Not confidence intervals.", fontsize=9)
    save(fig, directory, "network_metrics")
    # Re-render both panels independently; do not crop or rescale the old PDF.
    for column, filename, title in [("frequency_auc", "frequency_auc", "Half-max frequency AUC"),
                                    ("frequency_effect", "frequency_effect", "Population frequency: CMV+ minus CMV−")]:
        fig, ax = plt.subplots(figsize=(9.0, 3.4), layout="constrained")
        boxes(ax, [frequency_values(metrics, k, column, common) for k in MODELS], SHORT_LABELS, colors,
              zero=column == "frequency_effect", limits=(-.03, 1.03) if column == "frequency_auc" else None)
        ax.set_title(title)
        ax.set_ylabel("Percentage points" if column == "frequency_effect" else "ROC-AUC")
        fig.supxlabel(f"{len(common)} jointly evaluable splits. Boxes: Q1–Q3; line: median; dots: splits.", fontsize=9)
        save(fig, directory, filename)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), layout="constrained")
    differences = data["paired_differences"]
    for ax, column, title in zip(axes, ["network_auc_delta", "frequency_auc_delta"],
                                 ["Network AUC minus CellCNN", "Frequency AUC minus CellCNN"]):
        boxes(ax, [differences.loc[differences.model.eq(k), column] for k in list(MODELS)[1:]],
              SHORT_LABELS[1:], colors[1:], zero=True)
        ax.set_title(title)
    save(fig, directory, "paired_differences")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), layout="constrained", sharey=True)
    for ax, (model, (label, color)) in zip(axes, MODELS.items()):
        freq = data["frequencies"]
        points = donor_frequencies(freq.loc[freq.model.eq(model) & freq.split_id.isin(common)])
        rng = np.random.default_rng(611)
        for y, marker in [(0, "o"), (1, "^")]:
            group = points.loc[points.y_true.eq(y) & points.frequency.notna()]
            ax.scatter(y + rng.uniform(-.13, .13, len(group)), 100 * group.frequency, color=color, marker=marker, s=35)
            ax.plot([y-.18, y+.18], [100 * group.frequency.mean()]*2, color="#172B43", lw=2)
        ax.set_xticks([0, 1], ["CMV−", "CMV+"]); ax.set_title(label, fontsize=11); ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Mean test frequency per donor (%)")
    fig.supxlabel("One point per donor: average across available test visits in the common splits. The selected population may vary by split.", fontsize=9)
    save(fig, directory, "donor_frequencies")
    alphas = data["learned_alphas"]
    fig, ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    keys = ["quadratic_soft", "prototype_soft"]
    boxes(ax, [100 * alphas.loc[alphas.model.eq(k), "alpha"] for k in keys],
          ["Quadratic Soft-α", "Prototype Soft-α"], [MODELS[k][1] for k in keys])
    ax.axhline(1, color="#526579", ls="--", lw=1, label="Initialization: 1%")
    ax.set(ylabel="Learned pooling fraction α (%)", title="Alpha across all selected-model filters")
    ax.legend(fontsize=11); save(fig, directory, "learned_alphas")

def write_tables(data, common, directory):
    summary = data["summary"].set_index("model"); metrics = data["split_metrics"]
    names = {"cellcnn": "CellCNN", "quadratic_top1": r"Quad.\ Top-1\%",
             "quadratic_soft": r"Quad.\ Soft-$\alpha$", "prototype_soft": "Prototype"}
    for kind in ["network", "frequency"]:
        lines = [r"% Generated from the frozen benchmark; numerical values are unchanged.",
                 r"\begin{tabular}{@{}lrr@{}}\toprule",
                 r"Model & Mean & Median\\\midrule" if kind == "network" else r"Model & AUC & Effect*\\\midrule"]
        for key in MODELS:
            row = summary.loc[key]
            values = f"{row.network_auc_mean:.4f} & {row.network_auc_median:.4f}" if kind == "network" else f"{row.frequency_auc_mean:.4f} & {100 * row.frequency_effect_mean:.3f}"
            lines.append(names[key] + " & " + values + r"\\")
        lines.append(r"\bottomrule\end{tabular}")
        if kind == "frequency": lines.append(r"\par\smallskip{\tiny *Percentage points; common splits.}")
        (directory / f"{kind}_table.tex").write_text("\n".join(lines) + "\n")
    wide = metrics.pivot(index="split_id", columns="model", values="network_auc")
    delta = wide.quadratic_soft - wide.quadratic_top1
    vs_baseline = wide.quadratic_soft - wide.cellcnn
    alpha = data["learned_alphas"].groupby("model").alpha.agg(["median", "size"])
    macros = {
        "CommonFrequencySplits": str(len(common)),
        "NetworkTakeaway": f"Quadratic Top-1\\% and Soft-$\\alpha$ share the highest mean network AUC ({summary.loc['quadratic_top1','network_auc_mean']:.4f}).",
        "ClassificationTakeaway": f"Highest means: Quadratic Top-1\\% for AP ({summary.loc['quadratic_top1','average_precision_mean']:.3f}) and BA ({summary.loc['quadratic_top1','balanced_accuracy_mean']:.3f}).",
        "FrequencyTakeaway": f"Highest mean frequency AUC: Quadratic Top-1\\% ({summary.loc['quadratic_top1','frequency_auc_mean']:.4f}).",
        "EffectTakeaway": f"Largest mean frequency difference: CellCNN ({100 * summary.loc['cellcnn','frequency_effect_mean']:.3f} percentage points).",
        "PairedTakeaway": f"Quadratic Soft-$\\alpha$ vs.\\ CellCNN: {int(vs_baseline.gt(0).sum())} better, {int(vs_baseline.eq(0).sum())} tied, {int(vs_baseline.lt(0).sum())} worse; mean $\\Delta$AUC ${vs_baseline.mean():+.4f}$.",
        "AlphaTakeaway": f"Median $\\alpha$: Quadratic {100 * alpha.loc['quadratic_soft','median']:.2f}\\%; Prototype {100 * alpha.loc['prototype_soft','median']:.2f}\\%. Alpha controls pooling, not measured population frequency.",
        "HardSoftCounts": f"{int(delta.gt(0).sum())} better, {int(delta.eq(0).sum())} tied, {int(delta.lt(0).sum())} worse",
        "QuadraticFilterCount": str(int(alpha.loc['quadratic_soft','size'])),
        "PrototypeFilterCount": str(int(alpha.loc['prototype_soft','size'])),
        "FinalTakeaway": f"Quadratic Hard and Soft both reach {summary.loc['quadratic_top1','network_auc_mean']:.4f} mean network AUC; learnable soft pooling does not improve it here.",
    }
    assert abs(delta.mean()) < 1e-12, "Revisit the equality takeaway for an updated benchmark."
    assert summary.network_auc_mean.max() == summary.loc['quadratic_top1', 'network_auc_mean']
    assert summary.average_precision_mean.idxmax() == 'quadratic_top1'
    assert summary.balanced_accuracy_mean.idxmax() == 'quadratic_top1'
    assert summary.frequency_auc_mean.idxmax() == 'quadratic_top1'
    assert summary.frequency_effect_mean.idxmax() == 'cellcnn'
    text = "% Generated numerical claims; do not edit independently of the benchmark.\n"
    text += "\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in macros.items())
    (directory / "numbers.tex").write_text(text + "\n")
    return macros

def export(output=OUT):
    output = Path(output); data_dir = output / "data"; figures = output / "figures"
    data, common, manifest = load_data(data_dir)
    plot_assets(data, common, figures)
    macros = write_tables(data, common, data_dir)
    provenance = {
        "source_commit": manifest["source_commit"], "source_branch": manifest["source_branch"],
        "input_sha256": manifest["input_sha256"], "common_frequency_splits": sorted(map(int, common)),
        "language": "English", "model_training": False,
        "roc_aggregation": "Stored mean of interpolated split ROCs; legend uses mean original split AUC.",
        "frequency_effect": "CMV-positive minus CMV-negative donor mean frequency, multiplied by 100 for percentage points.",
        "donor_aggregation": "One point per donor: average over their test visits in the common frequency splits.",
        "takeaways": macros, "exporter_sha256": digest(__file__),
        "output_sha256": {str(p.relative_to(output)): digest(p) for p in sorted(figures.glob("*.pdf"))},
    }
    (data_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Exported English plots and slide tables to {output}; {len(common)} common frequency splits.")

if __name__ == "__main__":
    export()
