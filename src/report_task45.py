"""Report figures from the frozen Task 4/5 classifiers.

Citrus requires reconstruction of original final training trees, but no new
classifier fitting or hyperparameter selection. See task5_citrus_reconstruct.R.

python -m src.report_task45 --source-root results/tables/benchmark_main_20260912
python -m src.report_task45 --render-only
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "praesentation/report_assets/aufgaben_04_05"
MARKERS = ["CD3", "CD19", "CD56", "CD16", "CD94", "NKG2A", "NKG2C", "CD57"]
METHODS = ["CellCNN", "SVM", "Citrus"]
COLORS = ["#087f8c", "#6263aa", "#d98c30"]
METRICS = ["roc_auc", "average_precision", "balanced_accuracy"]


def positive_filter_union(responses, thresholds, contrasts):
    """One vote per cell/split, regardless of the number of positive filters.

    A zero reference maximum or nonpositive output contrast cannot select cells.
    Equality at the half-maximum threshold is excluded. Thresholds are never
    estimated on the cells being scored.
    """
    responses = np.asarray(responses)
    thresholds, contrasts = np.asarray(thresholds), np.asarray(contrasts)
    if (responses.ndim != 2 or thresholds.shape != (responses.shape[1],)
            or contrasts.shape != thresholds.shape
            or not all(np.isfinite(v).all() for v in [responses, thresholds, contrasts])
            or (thresholds < 0).any()):
        raise ValueError("Invalid responses, training thresholds or contrasts.")
    eligible = (thresholds > 0) & (contrasts > 0)
    return (responses[:, eligible] > thresholds[eligible]).any(axis=1)


def profile_summaries(visits, markers, mean, scale):
    """Cells -> nonempty test visits -> donors; every contributing donor equal.

    An empty selection has no marker profile, rather than a zero-valued profile.
    Its visit is retained in the visit table and in cell-frequency denominators.
    """
    keys = ["method", "donor_id"]
    total = visits.groupby(keys).size().rename("test_visits")
    valid = visits.loc[visits.n_selected.gt(0)]
    donor = valid.groupby(keys)[markers].mean()
    donor = total.to_frame().join(donor).join(
        valid.groupby(keys).size().rename("nonempty_visits")).reset_index()
    donor["nonempty_visits"] = donor.nonempty_visits.fillna(0).astype(int)
    records = []
    for method, frame in donor.groupby("method", sort=False):
        z = (frame[markers].to_numpy() - np.asarray(mean)) / np.asarray(scale)
        for j, marker in enumerate(markers):
            values = z[:, j][np.isfinite(z[:, j])]
            if not len(values):
                raise ValueError(f"No selected-cell profiles for {method}.")
            records.append(dict(method=method, marker=marker, mean_z=values.mean(),
                                median_z=np.median(values), q1_z=np.quantile(values, .25),
                                q3_z=np.quantile(values, .75), n_donors=len(values)))
    return donor, pd.DataFrame(records)


def group_profile_summaries(centroids, groups, markers, mean, scale):
    """Average within group/split, then describe variation across distinct splits."""
    kept = groups.loc[groups.retained]
    members = centroids.merge(kept[["method", "group_id"]], on=["method", "group_id"],
                              validate="many_to_one")
    split_profiles = members.groupby(["method", "group_id", "split_id"], as_index=False)[markers].mean()
    records = []
    for (method, group_id), frame in split_profiles.groupby(["method", "group_id"]):
        z = (frame[markers].to_numpy() - np.asarray(mean)) / np.asarray(scale)
        for j, marker in enumerate(markers):
            records.append(dict(method=method, group_id=group_id, marker=marker,
                                median_z=np.median(z[:, j]), q1_z=np.quantile(z[:, j], .25),
                                q3_z=np.quantile(z[:, j], .75), n_splits=len(frame)))
    return split_profiles, pd.DataFrame(records)


def performance_tables(artifacts):
    """Recompute donor metrics and align paired differences by split ID."""
    from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

    records = []
    for method in METHODS:
        for split_id, frame in artifacts[f"{method.lower()}_predictions"].groupby("split_id"):
            if len(frame) != 6 or frame.y_true.sum() != 2:
                raise ValueError("Expected six test donors (two positive) per split.")
            records.append(dict(method=method, split_id=split_id,
                                roc_auc=roc_auc_score(frame.y_true, frame.score),
                                average_precision=average_precision_score(frame.y_true, frame.score),
                                balanced_accuracy=balanced_accuracy_score(frame.y_true, frame.y_pred)))
    metrics = pd.DataFrame(records)
    wide = metrics.pivot(index="split_id", columns="method", values="roc_auc")
    if wide.isna().any().any():
        raise ValueError("Incomplete paired metrics.")
    pairs = [("CellCNN", "SVM"), ("CellCNN", "Citrus"), ("SVM", "Citrus")]
    differences = pd.concat([pd.DataFrame(dict(split_id=wide.index, first=a, second=b,
                                              auc_difference=(wide[a] - wide[b]).to_numpy()))
                             for a, b in pairs], ignore_index=True)
    summary = []
    for method, frame in metrics.groupby("method"):
        for metric in METRICS:
            v = frame[metric]
            summary.append(dict(method=method, metric=metric, n_splits=len(v),
                                mean=v.mean(), median=v.median(), q1=v.quantile(.25), q3=v.quantile(.75)))
    paired = []
    for (a, b), frame in differences.groupby(["first", "second"], sort=False):
        v = frame.auc_difference
        paired.append(dict(first=a, second=b, n_splits=len(v), mean=v.mean(), median=v.median(),
                           q1=v.quantile(.25), q3=v.quantile(.75),
                           better=int((v > 0).sum()), equal=int((v == 0).sum()), worse=int((v < 0).sum())))
    return metrics, pd.DataFrame(summary), differences, pd.DataFrame(paired)


def validate_citrus_input_readers(source_root, cache):
    """Check every raw marker value against native Citrus/FlowCore, byte for byte."""
    import hashlib
    import warnings
    import flowkit as fk
    from src.task5_citrus_oof import digest

    root = Path(source_root)
    data_root = root / "NK_cell_dataset/NK_cell_dataset"
    markers = pd.read_csv(data_root / "NK_markers.csv", header=None).iloc[0].dropna().tolist()
    native = pd.read_csv(Path(cache) / "native_input_checks.csv")
    donors = sorted(pd.read_csv(root / "results/tables/task2_cells.csv").sample_id.unique())
    if native.donor_id.duplicated().any() or sorted(native.donor_id) != donors:
        raise ValueError("Incomplete native Citrus input audit.")
    checked = []
    for row in native.itertuples(index=False):
        path = data_root / "NK_cell_dataset/gated_alive" / f"{row.donor_id}_alive.fcs"
        if digest(path, "md5") != row.source_fcs_md5:
            raise ValueError("Native Citrus input audit refers to different FCS data.")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"FCS file .* reported incorrect data offset.*")
            sample = fk.Sample(str(path), ignore_offset_error=True)
        values = sample.get_events(source="raw")[:, [sample.pns_labels.index(m) for m in markers]].astype("<f8")
        checksum = hashlib.md5(values.tobytes(order="F")).hexdigest()
        if values.shape != (row.n_cells, row.n_markers) or checksum != row.native_marker_input_md5:
            raise ValueError(f"FlowKit input differs from native Citrus/FlowCore: {row.donor_id}")
        checked.append(row._asdict() | dict(python_marker_input_md5=checksum, exact_match=True))
    return pd.DataFrame(checked)


def compute(source_root, output, citrus_cache):
    import torch
    from src.task5_interpretation import (
        load_artifacts, load_inputs, cellcnn_centroids, citrus_centroids, group_centroids,
        filter_responses, restore_scaler, svm_cell_selection, aggregate_svm_frequencies,
        file_hash, csv_read,
    )
    from src.task5_citrus_oof import evaluate_citrus

    source_root, output = Path(source_root), Path(output)
    tables = source_root / "results/tables"
    torch.set_num_threads(1)
    print("Validate saved models, splits and original inputs", flush=True)
    artifacts = load_artifacts(source_root)
    inputs = load_inputs(source_root)
    citrus_input_checks = validate_citrus_input_readers(source_root, citrus_cache)
    markers = inputs["markers"]
    print("Reconstruct training-only filter thresholds and subset groups", flush=True)
    cnn = cellcnn_centroids(inputs, artifacts)
    citrus = citrus_centroids(artifacts, markers)
    centroids, groups = group_centroids(pd.concat([cnn, citrus], ignore_index=True), markers,
                                       inputs["exploration"], artifacts["split_ids"])
    old_centroids = csv_read(tables / "task5_paper_centroids.csv")
    identity = ["method", "split_id", "subset_id"]
    a = centroids.set_index(identity).sort_index()
    b = old_centroids.set_index(identity).sort_index()
    pd.testing.assert_index_equal(a.index, b.index)
    np.testing.assert_allclose(a[markers], b[markers], atol=1e-12, rtol=0)
    np.testing.assert_allclose(a.response_threshold, b.response_threshold, atol=1e-12, rtol=0, equal_nan=True)
    pd.testing.assert_frame_equal(groups, csv_read(tables / "task5_paper_groups.csv"), check_dtype=False)

    visits, cell_records, thresholds = [], {m: [] for m in METHODS[:2]}, []
    for split_id in artifacts["split_ids"]:
        split = artifacts["splits"].loc[artifacts["splits"].split_id.eq(split_id)]
        filters = artifacts["cellcnn_filters"].loc[artifacts["cellcnn_filters"].split_id.eq(split_id)]
        meta = filters.groupby("filter_id", sort=True).first()
        # Float32 contrast matches the original model and centroid reconstruction.
        contrast = meta.output_weight_1.to_numpy(np.float32) - meta.output_weight_0.to_numpy(np.float32)
        threshold = cnn.loc[cnn.split_id.eq(split_id)].set_index("subset_id").response_threshold.reindex(meta.index).fillna(0).to_numpy()
        for fid, t, c in zip(meta.index, threshold, contrast):
            thresholds.append(dict(split_id=split_id, filter_id=fid, response_threshold=t,
                                   output_contrast=float(c), eligible=bool(t > 0 and c > 0)))
        svm = artifacts["svm_models"].loc[artifacts["svm_models"].split_id.eq(split_id)].set_index("marker").loc[markers]
        scaler = restore_scaler(svm)
        for donor in sorted(split.loc[split.outer_partition.eq("test"), "donor_id"]):
            values = inputs["data"][donor]
            scores = scaler.transform(values) @ svm.weight.to_numpy() + svm.intercept.iloc[0]
            svm_positive, _, top = svm_cell_selection(scores, float(svm.decision_threshold.iloc[0]))
            saved = artifacts["svm_predictions"].loc[
                artifacts["svm_predictions"].split_id.eq(split_id) & artifacts["svm_predictions"].donor_id.eq(donor)].iloc[0]
            np.testing.assert_allclose(scores[top].mean(), saved.score, atol=1e-10, rtol=0)
            cnn_positive = positive_filter_union(filter_responses(values, filters, markers), threshold, contrast)
            cells = inputs["cells"].loc[inputs["cells"].sample_id.eq(donor)]
            for method, selected in [("CellCNN", cnn_positive), ("SVM", svm_positive)]:
                count = int(selected.sum())
                profile = values[selected].mean(axis=0, dtype=np.float64) if count else np.full(len(markers), np.nan)
                visits.append(dict(method=method, split_id=split_id, donor_id=donor, y_true=int(saved.y_true),
                                   n_cells=len(values), n_selected=count, selected_fraction=count / len(values),
                                   **dict(zip(markers, profile))))
                cell_records[method].append(pd.DataFrame(dict(split_id=split_id, cell_id=cells.cell_id,
                                                              positive=selected[cells.event_index.to_numpy()], negative=False)))
        if (split_id + 1) % 5 == 0:
            print(f"Full held-out donors scored: {split_id + 1}/30 splits", flush=True)
    visits = pd.DataFrame(visits)
    frequencies = []
    for method, records in cell_records.items():
        frame = aggregate_svm_frequencies(pd.concat(records, ignore_index=True), inputs["cells"], artifacts["splits"])
        frame = frame.drop(columns=["negative_count", "negative_frequency"]).assign(method=method)
        frequencies.append(frame)
    frequencies = pd.concat(frequencies, ignore_index=True)
    old_svm = csv_read(tables / "task5_paper_svm_cells.csv").set_index("cell_id").sort_index()
    new_svm = frequencies.loc[frequencies.method.eq("SVM")].set_index("cell_id").sort_index()
    for column in ["n_test_models", "positive_count", "positive_frequency"]:
        np.testing.assert_allclose(new_svm[column], old_svm[column], atol=1e-14, rtol=0)
    print("Map full held-out donors to the reconstructed Citrus training clusters", flush=True)
    citrus_visits, citrus_frequencies, citrus_audit, citrus_hashes = evaluate_citrus(inputs, artifacts, citrus_cache)
    citrus_environments = [json.loads((Path(citrus_cache) / f"split_{sid:02d}/manifest.json").read_text())["package_versions"]
                           for sid in artifacts["split_ids"]]
    if len({json.dumps(env, sort_keys=True) for env in citrus_environments}) != 1:
        raise ValueError("Citrus trees were reconstructed in different package environments.")
    visits = pd.concat([visits, citrus_visits], ignore_index=True)
    frequencies = pd.concat([frequencies, citrus_frequencies], ignore_index=True)
    denominators = frequencies.pivot(index="cell_id", columns="method", values="n_test_models")
    for method in METHODS[1:]:
        np.testing.assert_array_equal(denominators.CellCNN, denominators[method])
    if visits.duplicated(["method", "split_id", "donor_id"]).any() or len(visits) != 540:
        raise ValueError("Incomplete or duplicated held-out donor visits.")
    mu, scale = inputs["exploration"]["scaler_mean"], inputs["exploration"]["scaler_scale"]
    donor_profiles, oof_profiles = profile_summaries(visits, markers, mu, scale)
    group_splits, group_profiles = group_profile_summaries(centroids, groups, markers, mu, scale)
    metrics, metric_summary, differences, paired_summary = performance_tables(artifacts)
    old_metrics = csv_read(tables / "task4_split_metrics.csv").rename(columns={"method_label": "method"})
    old_metrics["method"] = old_metrics.method.replace({"Lineare Single-Cell-SVM": "SVM"})
    old_metrics = old_metrics.set_index(["method", "split_id"]).sort_index()
    np.testing.assert_allclose(metrics.set_index(["method", "split_id"]).sort_index()[METRICS],
                               old_metrics[METRICS], atol=1e-14, rtol=0)

    frames = dict(split_metrics=metrics, metric_summary=metric_summary, paired_differences=differences,
                  paired_summary=paired_summary, cell_selection_frequencies=frequencies,
                  profiles_by_test_visit=visits, profiles_by_donor=donor_profiles,
                  oof_marker_profiles=oof_profiles, groups=groups, group_profiles_by_split=group_splits,
                  group_marker_profiles=group_profiles, cellcnn_filter_thresholds=pd.DataFrame(thresholds),
                  citrus_mapping_audit=citrus_audit, citrus_input_checks=citrus_input_checks)
    directory = output / "data"
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    stats = {}
    for method in METHODS:
        f = frequencies.loc[frequencies.method.eq(method)]
        v = visits.loc[visits.method.eq(method)]
        stats[method] = dict(map_cells=len(f), selected_at_least_once=int(f.positive_count.gt(0).sum()),
                             selected_at_least_half=int(f.positive_frequency.ge(.5).sum()),
                             selected_every_test_visit=int(f.positive_frequency.eq(1).sum()),
                             nonempty_visits=int(v.n_selected.gt(0).sum()), test_visits=len(v),
                             min_test_visits=int(f.n_test_models.min()), max_test_visits=int(f.n_test_models.max()),
                             contributing_donors=int(donor_profiles.loc[donor_profiles.method.eq(method)].nonempty_visits.gt(0).sum()))
    input_names = ["task2_cells.csv", "task2_embeddings.csv", "task2_provenance.json",
                   "task5_paper_centroids.csv", "task5_paper_groups.csv", "task5_paper_svm_cells.csv",
                   "task5_paper_provenance.json", "task4_split_metrics.csv"]
    provenance = dict(source_root=str(source_root.resolve()), split_ids=list(artifacts["split_ids"]),
                      gate="gated_alive", markers=markers, display_markers=MARKERS,
                      map="Existing gated_alive tsne_p30 reference: 10000 cells, 500 per donor; exploratory coordinates",
                      cellcnn_selection="Union across all positive-output-contrast filters per split; response strictly above training-reference half-maximum; zero-max filters excluded; outer test donors only",
                      svm_selection="Full donor top ceil(1% N) margins, then margin > saved inner-trained donor threshold; outer test donors only",
                      citrus_selection="Union of memberships of saved clusters with coefficient > 1e-10. Original final Ward training tree and sampled training cells reconstructed with original seed; exact nearest training cell in unscaled float64 arcsinh(raw/5) marker space; original training row order breaks ties. Only outer test donors, all their original events.",
                      denominator="All outer test visits of the cell's donor among splits 0-29, including visits with no selected cells",
                      oof_profiles="Mean selected full-donor cells per nonempty visit; mean nonempty visits per donor; equal donor weights. Empty selections have no marker profile. Mean and Q1/Q3 across donor profiles; not confidence intervals.",
                      group_profiles="Existing retrospective centroid grouping unchanged. Mean member centroids within group/split; median and Q1/Q3 across splits where group occurs. Absent groups have no profile. All coefficient signs retained; positive/negative recurrence reported separately.",
                      biological_scale="All 37 arcsinh(x/5) markers; common saved exploratory map z-scale for display only. Eight display markers as in the presentation.",
                      interpretation="Selection frequencies and group recurrence describe stability under overlapping donor splits, not causal cell effects or independent biological validation. Threshold rules differ between methods.",
                      checks=["Original model configurations and input hashes verified", "Training centroids, thresholds and groups reproduced", "SVM full-donor scores and original map frequencies reproduced", "All three methods share every test-cell denominator", "All Task 4 metrics independently recomputed from saved predictions", "Reconstructed Citrus selected training-cluster centroids match saved profiles", "384 native Citrus nearest-training-event queries checked per nonnull split", "All raw marker values in all 20 donors agree byte-for-byte between Citrus/FlowCore and Python/FlowKit"],
                      input_sha256=inputs["input_hashes"],
                      artifact_sha256=artifacts["source_hashes"] | {name: file_hash(tables / name) for name in input_names},
                      citrus_cache=str(Path(citrus_cache).resolve()), citrus_reconstruction_sha256=citrus_hashes,
                      citrus_reconstruction_package_versions=citrus_environments[0],
                      implementation_sha256={name: file_hash(ROOT / "src" / name) for name in ["report_task45.py", "task5_interpretation.py", "task4_artifacts.py", "task23_analysis.py", "task5_citrus_oof.py", "task5_citrus_reconstruct.R", "task5_citrus_input_check.R"]},
                      package_versions={name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "torch", "flowkit", "matplotlib"]},
                      results=stats, output_sha256={f"{name}.csv": file_hash(directory / f"{name}.csv") for name in frames})
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)
    return frames, provenance


def load_export(output):
    from src.task5_interpretation import csv_read, file_hash
    directory = Path(output) / "data"
    provenance = json.loads((directory / "provenance.json").read_text())
    frames = {}
    for name, expected in provenance["output_sha256"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Export checksum mismatch: {name}")
        frames[Path(name).stem] = csv_read(directory / name)
    return frames, provenance


def render(frames, provenance, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    directory = Path(output) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
                         "axes.labelcolor": "#243449", "text.color": "#243449"})
    overview = PdfPages(directory / "plot_overview.pdf", metadata={"CreationDate": None, "ModDate": None})
    main_figures = PdfPages(directory / "report_figures.pdf", metadata={"CreationDate": None, "ModDate": None})
    profile_limits = [frames["oof_marker_profiles"].loc[frames["oof_marker_profiles"].marker.isin(MARKERS), "mean_z"].abs().max(),
                      frames["group_marker_profiles"].loc[frames["group_marker_profiles"].marker.isin(MARKERS), "median_z"].abs().max()]
    marker_limit = max(1, int(np.ceil(max(profile_limits))))

    def display_value(value):
        return f"{0. if abs(value) < .05 else value:.1f}"

    def save(fig, name):
        fig.savefig(directory / f"{name}.pdf", dpi=400, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight")
        overview.savefig(fig, dpi=400, bbox_inches="tight")
        if name in ["task4_performance_paired", "task5_report_combined"]:
            main_figures.savefig(fig, dpi=400, bbox_inches="tight")
        plt.close(fig)

    def boxes(ax, arrays, labels, colors):
        box = ax.boxplot(arrays, positions=np.arange(len(arrays)), widths=.48, patch_artist=True, showfliers=False,
                         medianprops={"color": "#243449", "linewidth": 1.2})
        rng = np.random.default_rng(4531)
        for i, (v, color, patch) in enumerate(zip(arrays, colors, box["boxes"])):
            patch.set(facecolor=color, edgecolor=color, alpha=.22)
            ax.scatter(i + rng.uniform(-.13, .13, len(v)), v, color=color, alpha=.65, s=12, linewidths=0)
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.grid(axis="y", alpha=.15)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]})
    metrics = frames["split_metrics"]
    boxes(axes[0], [metrics.loc[metrics.method.eq(m), "roc_auc"].to_numpy() for m in METHODS], METHODS, COLORS)
    axes[0].axhline(.5, color="#7d8996", ls="--", lw=.7)
    axes[0].set(title="A  Classification performance", ylabel="Test ROC-AUC", ylim=(-.04, 1.04))
    differences, paired = frames["paired_differences"], frames["paired_summary"]
    arrays, labels = [], []
    for row in paired.itertuples():
        arrays.append(differences.loc[differences["first"].eq(row.first) & differences.second.eq(row.second), "auc_difference"].to_numpy())
        labels.append(f"{row.first} − {row.second}\n{row.better} / {row.equal} / {row.worse}")
    boxes(axes[1], arrays, labels, [COLORS[0], COLORS[0], COLORS[1]])
    axes[1].axhline(0, color="#7d8996", ls="--", lw=.8)
    axes[1].set(title="B  Paired differences on the same splits", ylabel="ROC-AUC difference")
    axes[1].tick_params(axis="x", labelsize=7)
    fig.supxlabel("30 shared splits of 20 donors. Boxes: Q1–Q3; line: median; dots: splits.\nPaired counts: better / equal / worse. Split variation is descriptive, not a confidence interval.", fontsize=7)
    save(fig, "task4_performance_paired")

    def maps(fig, axes):
        freq = frames["cell_selection_frequencies"]
        for index, (ax, method) in enumerate(zip(axes, METHODS)):
            frame = freq.loc[freq.method.eq(method)]
            ax.scatter(frame.component_1, frame.component_2, s=1.5, color="#dedfe3", linewidths=0, rasterized=True)
            selected = frame.loc[frame.positive_frequency.gt(0)].sort_values("positive_frequency")
            ax.scatter(selected.component_1, selected.component_2, c=selected.positive_frequency,
                       vmin=0, vmax=1, cmap="viridis", s=7, linewidths=0, rasterized=True)
            ax.set(title=f"{chr(65 + index)}  {method}", aspect="equal", xticks=[], yticks=[], xlabel="t-SNE 1",
                   ylabel="t-SNE 2" if index == 0 else "")
            for spine in ax.spines.values():
                spine.set_visible(False)
            stats = provenance["results"][method]
            ax.set_xlabel(f"≥1 visit: {stats['selected_at_least_once']}; ≥50%: {stats['selected_at_least_half']}", fontsize=7)
        fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="viridis"), ax=list(axes),
                     label="Positive selection frequency", ticks=[0, .5, 1], shrink=.85, pad=.015, fraction=.025)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), layout="constrained")
    maps(fig, axes)
    fig.supxlabel("All methods: outer test donors only, splits 0–29; same 10,000 reference cells (500/donor).\nDenominator: all 4–16 test visits per donor. Grey: never selected. Method-specific selection rules.", fontsize=7)
    save(fig, "task5_oof_selection")

    def oof_heatmap(fig, ax):
        profile = frames["oof_marker_profiles"]
        values = profile.pivot(index="method", columns="marker", values="mean_z").loc[METHODS, MARKERS].to_numpy()
        limit = marker_limit
        im = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(MARKERS)), MARKERS)
        ax.set_yticks(range(len(METHODS)), METHODS)
        ax.tick_params(length=0)
        for i, j in np.ndindex(values.shape):
            ax.text(j, i, display_value(values[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if abs(values[i, j]) > .65 * limit else "#243449")
        ax.set_title("D  Selected test-cell profiles · equal donor weights", loc="left")
        fig.colorbar(im, ax=ax, label="Mean marker z-score", shrink=.95, pad=.02)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.25), layout="constrained", sharex=True)
    profile = frames["oof_marker_profiles"]
    for ax, method, color in zip(axes, METHODS, COLORS):
        p = profile.loc[profile.method.eq(method)].set_index("marker").loc[MARKERS]
        y = np.arange(len(MARKERS))
        ax.hlines(y, p.q1_z, p.q3_z, color=color, lw=3, alpha=.5)
        ax.scatter(p.mean_z, y, s=23, color=color, zorder=3, label="Mean across donors")
        ax.axvline(0, color="#7d8996", ls="--", lw=.7)
        ax.set(yticks=y, yticklabels=MARKERS, title=f"{method}: {int(p.n_donors.iloc[0])} donors", xlabel="Marker z-score")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
    fig.supxlabel("Mean selected cells per nonempty test visit → mean visits per donor → equal donor weights.\nDots: mean across donors; lines: Q1–Q3 across donor profiles, not confidence intervals. Full donor selections; same eight markers.", fontsize=7)
    save(fig, "task5_oof_marker_profiles")

    def group_heatmap(fig, axes):
        groups = frames["groups"].loc[frames["groups"].retained].sort_values(["method", "group_id"])
        profiles = frames["group_marker_profiles"].pivot(index=["method", "group_id"], columns="marker", values="median_z")
        idx = pd.MultiIndex.from_frame(groups[["method", "group_id"]])
        values = profiles.loc[idx, MARKERS].to_numpy()
        limit = marker_limit
        im = axes[0].imshow(values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        axes[0].set_xticks(range(len(MARKERS)), MARKERS)
        axes[0].set_yticks(range(len(groups)), [f"{r.method} G{r.group_id}" for r in groups.itertuples()])
        axes[0].tick_params(length=0, labelsize=7)
        for i, j in np.ndindex(values.shape):
            axes[0].text(j, i, display_value(values[i, j]), ha="center", va="center", fontsize=6.5,
                         color="white" if abs(values[i, j]) > .65 * limit else "#243449")
        axes[0].set_title("Marker profiles across recurring groups", loc="left")
        y = np.arange(len(groups))
        axes[1].barh(y, groups.occurrences, color=[COLORS[METHODS.index(m)] for m in groups.method], alpha=.7, height=.65)
        axes[1].set(xlim=(0, 51), ylim=(len(groups)-.5, -.5), yticks=[], xticks=[0, 15, 30], xlabel="Distinct splits (of 30)")
        axes[1].axvline(30, color="#7d8996", ls=":", lw=.6)
        axes[1].set_title("Recurrence  |  + / −", loc="left")
        for i, row in enumerate(groups.itertuples()):
            axes[1].text(row.occurrences + .5, i, str(row.occurrences), va="center", fontsize=7)
            axes[1].text(39, i, f"{row.positive_splits} / {row.negative_splits}", va="center", fontsize=7)
        fig.colorbar(im, ax=axes[0], label="Median marker z-score", pad=.015, fraction=.035)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4), layout="constrained", gridspec_kw={"width_ratios": [2.4, 1]})
    group_heatmap(fig, axes)
    fig.supxlabel("Groups present in ≥6/30 splits. Average member centroids within each split, then median across splits with that group.\nTraining-subset profiles; + / −: splits with positive / negative members (can overlap). Group absence is not zero expression.", fontsize=7)
    save(fig, "task5_group_profiles_recurrence")

    fig = plt.figure(figsize=(6.4, 3.35), layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.1])
    top = grid[0].subgridspec(1, 3)
    maps(fig, [fig.add_subplot(top[i]) for i in range(3)])
    oof_heatmap(fig, fig.add_subplot(grid[1]))
    fig.supxlabel("30 shared splits; only held-out donors. Maps: the same 10,000 cells; colour: selection / donor's 4–16 test visits.\nProfiles: selected full-donor test cells → nonempty visits → equal donor weights. Grey: never selected; rules differ by method.", fontsize=7)
    save(fig, "task5_report_combined")
    overview.close()
    main_figures.close()
    from src.task5_citrus_oof import digest
    figure_manifest = dict(renderer_sha256=digest(__file__),
                           input_provenance_sha256=digest(Path(output) / "data/provenance.json"),
                           main_figures=["task4_performance_paired.pdf", "task5_report_combined.pdf"],
                           output_sha256={path.name: digest(path) for path in sorted(directory.iterdir())
                                          if path.suffix in [".pdf", ".png"]})
    (directory / "provenance.json").write_text(json.dumps(figure_manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "results/tables/benchmark_main_20260912")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--citrus-cache", type=Path, default=ROOT / "results/tables/task5_citrus_oof_cache")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if not args.render_only:
        missing = [sid for sid in range(30) if not (args.citrus_cache / f"split_{sid:02d}/manifest.json").exists()]
        if missing or not (args.citrus_cache / "native_input_checks.csv").exists():
            parser.error(f"Citrus reconstructions/input audit missing (splits {missing}). Run src/task5_citrus_reconstruct.R and src/task5_citrus_input_check.R first; see the report-assets README.")
    frames, provenance = load_export(args.output) if args.render_only else compute(args.source_root, args.output, args.citrus_cache)
    render(frames, provenance, args.output)
    print(f"Report assets: {args.output}", flush=True)


if __name__ == "__main__":
    main()
