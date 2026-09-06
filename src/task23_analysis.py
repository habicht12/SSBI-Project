"""Kleine gemeinsame Bausteine für die rein explorativen Aufgaben 2 und 3."""

import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SEED = 42
CELLS_PER_DONOR = 500
COFACTOR = 5.0
BASELINE_VARIANTS = {"PCA": "pca", "t-SNE": "tsne_p30", "UMAP": "umap_n15_d0.1"}


def projection_configs():
    """Festes kleines Sensitivitätsraster; jede Konfiguration ist eigenständig."""
    configs = []
    for whiten in (False, True):
        configs.append(dict(variant="pca_white" if whiten else "pca", method="PCA",
                            params=dict(n_components=2, whiten=whiten, svd_solver="full")))
    for perplexity in (5, 30, 50):
        configs.append(dict(variant=f"tsne_p{perplexity}", method="t-SNE",
                            params=dict(n_components=2, perplexity=perplexity, init="pca",
                                        learning_rate="auto", max_iter=1000, early_exaggeration=12,
                                        metric="euclidean", method="barnes_hut", angle=0.5,
                                        random_state=SEED, n_jobs=1)))
    for neighbors, min_dist in ((15, 0.1), (50, 0.1), (15, 0.5)):
        configs.append(dict(variant=f"umap_n{neighbors}_d{min_dist}", method="UMAP",
                            params=dict(n_components=2, n_neighbors=neighbors, min_dist=min_dist,
                                        metric="euclidean", init="spectral", n_epochs=500,
                                        random_state=SEED, n_jobs=1)))
    return configs


def sample_event_indices(event_count, donor_index, cells_per_donor=CELLS_PER_DONOR):
    """Originale nullbasierte Eventindizes, ohne Zurücklegen und stabil sortiert."""
    if not 0 < cells_per_donor <= event_count:
        raise ValueError("Jeder Spender muss mindestens die geforderte Zellzahl enthalten.")
    rng = np.random.default_rng(SEED + donor_index)
    return np.sort(rng.choice(event_count, size=cells_per_donor, replace=False))


def load_exploratory_data(project_root, cells_per_donor=CELLS_PER_DONOR):
    """Lese gated_alive; fitte ausschließlich den getrennten Explorations-Scaler.

    Alle 20 Spender werden ausdrücklich nur für Aufgaben 2/3 explorativ verwendet.
    Rückgabe: Zellmetadaten, arcsinh-Markerwerte, z-Werte, Scaler und Provenienz.
    """
    import flowkit as fk

    data_root = Path(project_root) / "NK_cell_dataset" / "NK_cell_dataset"
    labels = pd.read_csv(data_root / "NK_fcs_samples_with_labels.csv")
    labels["sample_id"] = labels["fcs_filename"].str.replace("_NK.fcs", "", regex=False)
    labels = labels.sort_values("sample_id").reset_index(drop=True)
    markers = pd.read_csv(data_root / "NK_markers.csv", header=None).iloc[0].dropna().tolist()
    if len(labels) != 20 or not labels["sample_id"].is_unique:
        raise ValueError("Erwartet werden genau 20 verschiedene Spender.")
    if len(markers) != 37 or len(set(markers)) != 37:
        raise ValueError("Erwartet werden genau 37 eindeutige Analysemarker.")
    metadata, values = [], []
    for donor_index, row in labels.iterrows():
        path = data_root / "NK_cell_dataset" / "gated_alive" / f"{row.sample_id}_alive.fcs"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"FCS file .* reported incorrect data offset.*")
            sample = fk.Sample(str(path), ignore_offset_error=True)
        if len(set(sample.pns_labels)) != len(sample.pns_labels):
            raise ValueError(f"Nicht eindeutige PnS-Kanäle: {path.name}")
        channel_indices = [sample.pns_labels.index(marker) for marker in markers]
        indices = sample_event_indices(sample.event_count, donor_index, cells_per_donor)
        raw = sample.get_events(source="raw")[np.ix_(indices, channel_indices)].astype(np.float64)
        if not np.isfinite(raw).all():
            raise ValueError(f"Nicht endliche Markerwerte: {path.name}")
        values.append(raw)
        metadata.append(pd.DataFrame({
            "cell_id": [f"gated_alive:{row.sample_id}:{i}" for i in indices],
            "sample_id": row.sample_id, "event_index": indices,
        }))
    cells = pd.concat(metadata, ignore_index=True)
    raw = np.concatenate(values)
    transformed = pd.DataFrame(np.arcsinh(raw / COFACTOR), columns=markers)
    scaler = StandardScaler().fit(transformed)
    scaled = scaler.transform(transformed)
    digest = hashlib.sha256()
    digest.update(json.dumps({"cell_ids": cells.cell_id.tolist(), "markers": markers},
                             separators=(",", ":")).encode())
    digest.update(raw.astype("<f8").tobytes())
    provenance = dict(
        gate="gated_alive", donors=labels.sample_id.tolist(), cells_per_donor=cells_per_donor,
        seed=SEED, sampling="default_rng(seed + sorted_donor_index), without replacement, sorted event indices",
        event_index_base=0, markers=markers, transformation="arcsinh(x / 5)",
        scaler_fit="exploratory sample of all 20 donors; not a benchmark transformer",
        scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
        selected_data_sha256=digest.hexdigest(),
    )
    return cells, transformed, scaled, scaler, provenance


def fit_projection(scaled, config):
    """Fitte eine Karte; Liefere Koordinaten und wenige relevante Fitdiagnosen."""
    if config["method"] == "PCA":
        model = PCA(**config["params"])
    elif config["method"] == "t-SNE":
        model = TSNE(**config["params"])
    elif config["method"] == "UMAP":
        from umap import UMAP
        model = UMAP(**config["params"])
    else:
        raise ValueError(f"Unbekannte Methode: {config['method']}")
    coordinates = model.fit_transform(scaled)
    if coordinates.shape != (len(scaled), 2) or not np.isfinite(coordinates).all():
        raise ValueError("Projektion liefert keine endliche zweidimensionale Karte.")
    diagnostics = {}
    if config["method"] == "PCA":
        diagnostics["explained_variance_ratio"] = model.explained_variance_ratio_.tolist()
    elif config["method"] == "t-SNE":
        diagnostics.update(kl_divergence=float(model.kl_divergence_), iterations=int(model.n_iter_))
    return coordinates, diagnostics


def neighbor_indices(values, k=15):
    """Exakte euklidische Nachbarn; eigene Zellidentität auch bei Duplikaten entfernen.

    Zeilenreihenfolge ist die gemeinsame Zellidentität aller verglichenen Räume.
    Distanzgleichstände folgen der deterministischen Bibliotheksreihenfolge.
    """
    if not 0 < k < len(values):
        raise ValueError("Nachbarzahl muss zwischen 1 und n_cells - 1 liegen.")
    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="brute", n_jobs=1)
    candidates = model.fit(values).kneighbors(values, return_distance=False)
    return np.array([row[row != i][:k] for i, row in enumerate(candidates)], dtype=np.int64)


def neighbor_jaccard(first, second, donor_ids):
    """Jaccard je Zelle, dann Mittel je Spender und gleich gewichtetes Spendermittel."""
    if first.shape != second.shape or len(donor_ids) != len(first):
        raise ValueError("Nachbarschaften benötigen gleiche Form und identische Zellreihenfolge.")
    overlap = (first[:, :, None] == second[:, None, :]).any(axis=2).sum(axis=1)
    scores = overlap / (2 * first.shape[1] - overlap)
    donor_scores = pd.Series(scores).groupby(np.asarray(donor_ids)).mean()
    return float(donor_scores.mean())


def recommend_projections(summary):
    """Bestes untersuchtes 37D-Jaccard je Verfahren; Gleichstand bevorzugt Baseline."""
    recommendations = []
    for method, baseline in BASELINE_VARIANTS.items():
        group = summary.loc[summary.method == method]
        best = group.reference_jaccard.max()
        tied = group.loc[np.isclose(group.reference_jaccard, best, rtol=0, atol=1e-12)]
        if tied.empty or not np.isfinite(best):
            raise ValueError(f"Keine gültige Referenzmetrik für {method}.")
        preferred = tied.loc[tied.variant == baseline]
        row = (preferred if len(preferred) else tied).iloc[0]
        recommendations.append({
            "method": method, "variant": row.variant, "reference_jaccard": row.reference_jaccard,
            "reason": ("Gleichstand innerhalb 1e-12: Baseline, danach Rasterreihenfolge."
                       if len(tied) > 1 else "Höchster 37D-Nachbarschaftserhalt im untersuchten Raster."),
        })
    return pd.DataFrame(recommendations)


def fit_clusterings(scaled):
    """Neun Partitionen im 37D-Raum; Ward-Baum und Leiden-Graph jeweils einmal."""
    from scipy.cluster.hierarchy import linkage, cut_tree
    from sklearn.cluster import KMeans
    import leidenalg

    partitions, configurations = {}, []
    for count in (6, 10, 14):
        variant = f"kmeans_k{count}"
        params = dict(n_clusters=count, n_init=20, max_iter=300, random_state=SEED)
        partitions[variant] = KMeans(**params).fit_predict(scaled)
        configurations.append(dict(variant=variant, method="K-Means", params=params))
    tree = linkage(scaled, method="ward", metric="euclidean")
    cuts = cut_tree(tree, n_clusters=[6, 10, 14])
    for column, count in enumerate((6, 10, 14)):
        variant = f"ward_k{count}"
        partitions[variant] = cuts[:, column]
        configurations.append(dict(variant=variant, method="Ward",
                                   params=dict(n_clusters=count, linkage="ward", metric="euclidean")))
    graph = make_neighbor_graph(scaled)
    for resolution in (0.5, 1.0, 1.5):
        variant = f"leiden_r{resolution}"
        partition = leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution, seed=SEED, n_iterations=-1,
        )
        partitions[variant] = np.asarray(partition.membership)
        configurations.append(dict(variant=variant, method="Leiden", params=dict(
            resolution=resolution, neighbors=15, graph="unweighted undirected kNN union",
            partition="RBConfigurationVertexPartition", seed=SEED, n_iterations=-1)))
    return partitions, configurations


def make_neighbor_graph(scaled, k=15):
    """Ungerichteter ungewichteter Uniongraph; jede Kante einmal, keine Selbstkanten."""
    import igraph as ig

    neighbors = neighbor_indices(scaled, k=k)
    edges = np.column_stack((np.repeat(np.arange(len(scaled)), k), neighbors.ravel()))
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    return ig.Graph(n=len(scaled), edges=edges.tolist(), directed=False)


def silhouette_evaluation(scaled, cells, partitions, cells_per_donor=100):
    """Gemeinsame approximative Silhouetten; fehlende Cluster lösen Gesamtauswertung aus."""
    from sklearn import config_context
    from sklearn.metrics import silhouette_samples

    indices = []
    donors = cells.sample_id.to_numpy()
    for donor_index, donor in enumerate(sorted(set(donors))):
        candidates = np.flatnonzero(donors == donor)
        if len(candidates) < cells_per_donor:
            raise ValueError("Zu wenige Zellen für die spenderbalancierte Silhouettenstichprobe.")
        indices.extend(np.random.default_rng(SEED + 1 + donor_index).choice(
            candidates, cells_per_donor, replace=False))
    indices = np.sort(indices)
    missing_clusters = any(len(np.unique(labels[indices])) < len(np.unique(labels))
                           for labels in partitions.values())
    if missing_clusters:
        indices = np.arange(len(cells))
    rows, donor_rows = [], []
    for variant, labels in partitions.items():
        _, sizes = np.unique(labels, return_counts=True)
        selected_labels = labels[indices]
        cluster_count = len(np.unique(selected_labels))
        valid = 2 <= cluster_count < len(indices)
        reason = "" if valid else "Silhouette nur für 2 bis n-1 vertretene Cluster definiert."
        if valid:
            with config_context(working_memory=256):
                scores = silhouette_samples(scaled[indices], selected_labels, metric="euclidean")
            by_donor = pd.Series(scores).groupby(donors[indices]).mean()
            mean_score = float(by_donor.mean())
            donor_rows.extend(dict(variant=variant, sample_id=donor, silhouette=float(score))
                              for donor, score in by_donor.items())
        else:
            mean_score = np.nan
        rows.append(dict(variant=variant, n_clusters=len(sizes), min_cluster_size=int(sizes.min()),
                         singleton_clusters=int((sizes == 1).sum()), evaluation_cells=len(indices),
                         silhouette=mean_score, invalid_reason=reason))
    return pd.DataFrame(rows), pd.DataFrame(donor_rows), indices, missing_clusters


def select_clusterings(summary):
    """Je Methode höchste Silhouette; bei Gleichstand weniger Cluster, dann Rasterfolge."""
    selected = []
    for method, group in summary.groupby("method", sort=False):
        valid = group.loc[np.isfinite(group.silhouette)]
        if valid.empty:
            raise ValueError(f"Keine gültige Clusterlösung für {method}.")
        tied = valid.loc[np.isclose(valid.silhouette, valid.silhouette.max(), rtol=0, atol=1e-12)]
        row = tied.sort_values("n_clusters", kind="stable").iloc[0]
        selected.append(row.variant)
    return selected


def cluster_profiles(transformed, cells, labels):
    """Median je Spender/Cluster, dann Median über vertretene Spender; keine Nullimputation."""
    frame = transformed.copy()
    frame["sample_id"] = cells.sample_id.to_numpy()
    frame["cluster"] = labels
    donor_profiles = frame.groupby(["cluster", "sample_id"])[transformed.columns].median()
    profiles = donor_profiles.groupby("cluster").median()
    donor_counts = frame.groupby(["cluster", "sample_id"]).size().unstack(fill_value=0)
    counts = pd.DataFrame({
        "cell_count": donor_counts.sum(axis=1),
        "donor_coverage": (donor_counts > 0).sum(axis=1),
        "dominant_donor_share": donor_counts.max(axis=1) / donor_counts.sum(axis=1),
    })
    return profiles, counts, donor_profiles, donor_counts
