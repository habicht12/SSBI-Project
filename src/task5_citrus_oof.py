"""Original Citrus training-cell mapping for held-out cells, with native audits."""

import hashlib
import json
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def digest(path, algorithm="sha256"):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def exact_nearest(tree, reference, query, workers=8):
    """Exact Euclidean nearest training event, first original row for ties.

    Citrus assigns to a training *cell*, not a cluster centroid. KD-tree eps=0
    accelerates the same Euclidean search. Near ties are recalculated with
    sequential float64 additions in marker order, as in native Citrus.
    """
    reference, query = np.asarray(reference), np.asarray(query)
    if (reference.ndim != 2 or query.ndim != 2 or query.shape[1] != reference.shape[1]
            or len(reference) < 2 or not np.isfinite(query).all()):
        raise ValueError("Invalid nearest-training-cell inputs.")
    distances, indices = tree.query(query, k=2, eps=0, workers=workers)
    nearest = indices[:, 0].copy()
    tolerance = 1e-12 * (1 + distances[:, 1])
    ambiguous = np.flatnonzero(distances[:, 1] - distances[:, 0] <= tolerance)
    for row in ambiguous:
        candidates = np.sort(tree.query_ball_point(query[row], distances[row, 1] + tolerance[row]))
        delta = reference[candidates] - query[row]
        square_distance = np.zeros(len(candidates), dtype=np.float64)
        for marker in range(query.shape[1]):
            square_distance += delta[:, marker] ** 2
        nearest[row] = candidates[np.sqrt(square_distance).argmin()]
    return nearest


def exact_positive_membership(tree, reference, positive, query, positive_tree=None, negative_tree=None, workers=6):
    """Exact union membership, avoiding unnecessary full nearest-cell searches.

    A real negative training cell closer than the closest positive cell proves
    nonselection. An approximate negative query may supply that witness; it
    never decides a positive selection. All unproven cases fall back to exact
    full-tree search, including near ties. Thus approximation affects speed,
    not the selection rule or the result.
    """
    positive = np.asarray(positive, dtype=bool)
    if not positive.any():
        return np.zeros(len(query), dtype=bool)
    if positive.all():
        return np.ones(len(query), dtype=bool)
    if positive_tree is None:
        positive_tree = cKDTree(reference[positive])
    if negative_tree is None:
        negative_tree = cKDTree(reference[~positive])
    positive_distance, _ = positive_tree.query(query, k=1, eps=0, workers=workers)
    negative_distance, _ = negative_tree.query(query, k=1, eps=2, workers=workers)
    proven_negative = negative_distance < positive_distance - 1e-10 * (1 + positive_distance)
    selected = np.zeros(len(query), dtype=bool)
    unresolved = ~proven_negative
    if unresolved.any():
        selected[unresolved] = positive[exact_nearest(tree, reference, query[unresolved], workers)]
    return selected


def load_cache(directory, source_root, split_id, markers, donor_splits):
    """Reject incomplete/stale reconstructions and any train/test mismatch."""
    directory, source_root = Path(directory), Path(source_root)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["split_id"] != split_id or manifest["markers"] != list(markers):
        raise ValueError("Citrus cache split or marker order mismatch.")
    split = donor_splits.loc[donor_splits.split_id.eq(split_id)]
    for partition in ["train", "test"]:
        expected = sorted(split.loc[split.outer_partition.eq(partition), "donor_id"])
        if manifest[f"{partition}_donors"] != expected:
            raise ValueError("Citrus cache donor partition mismatch.")
    for name, expected in manifest["fingerprint"].items():
        path = Path(__file__).with_name("task5_citrus_reconstruct.R") if name == "exporter_md5" else source_root / "results/tables" / name
        if digest(path, "md5") != expected:
            raise ValueError(f"Stale Citrus reconstruction source: {name}")
    for name, expected in (manifest["output_md5"] or {}).items():
        if digest(directory / name, "md5") != expected:
            raise ValueError(f"Citrus reconstruction output changed: {name}")
    if manifest["null_positive_selection"]:
        return manifest, None, None
    reference = np.fromfile(directory / "training.f64", dtype="<f8").reshape((manifest["n_training_cells"], len(markers)), order="F")
    positive = np.fromfile(directory / "positive.u8", dtype=np.uint8)
    if len(positive) != len(reference) or not np.isin(positive, [0, 1]).all() or not np.isfinite(reference).all():
        raise ValueError("Invalid reconstructed Citrus training values or membership.")
    return manifest, np.ascontiguousarray(reference), positive.astype(bool)


def load_full_values(root, markers, donors):
    """Citrus' original float64 arcsinh(raw/5), preserving FCS event order."""
    import flowkit as fk
    values = {}
    for donor in donors:
        path = Path(root) / "NK_cell_dataset/NK_cell_dataset/NK_cell_dataset/gated_alive" / f"{donor}_alive.fcs"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"FCS file .* reported incorrect data offset.*")
            sample = fk.Sample(str(path), ignore_offset_error=True)
        channels = [sample.pns_labels.index(marker) for marker in markers]
        values[donor] = np.arcsinh(sample.get_events(source="raw")[:, channels].astype(np.float64) / 5.)
        if not np.isfinite(values[donor]).all():
            raise ValueError(f"Nonfinite Citrus input: {donor}")
    return values


def evaluate_citrus(inputs, artifacts, cache, workers=6, values=None):
    """Full held-out donors, union of positive-coefficient cluster memberships."""
    from src.task5_interpretation import aggregate_svm_frequencies

    markers, cells = inputs["markers"], inputs["cells"]
    if values is None:
        values = load_full_values(inputs["root"], markers, sorted(cells.sample_id.unique()))
    visits, cell_records, audit = [], [], []
    source_hashes = {}
    for sid in artifacts["split_ids"]:
        directory = Path(cache) / f"split_{sid:02d}"
        manifest, reference, positive = load_cache(directory, inputs["root"], sid, markers, artifacts["splits"])
        source_hashes[f"split_{sid:02d}/manifest.json"] = digest(directory / "manifest.json")
        split = artifacts["splits"].loc[artifacts["splits"].split_id.eq(sid)]
        expected_positive = artifacts["citrus_clusters"].loc[
            artifacts["citrus_clusters"].split_id.eq(sid) & artifacts["citrus_clusters"].coefficient.gt(1e-10), "cluster_id"].unique()
        if set(manifest["positive_cluster_ids"]) != set(expected_positive):
            raise ValueError("Cached positive clusters disagree with saved classifier.")
        # A full-donor nearest-neighbour pass is expensive; resume only exact,
        # checksum-verified computations for this code, input data and tree.
        fingerprint = dict(implementation=digest(__file__), reconstruction=digest(directory / "manifest.json"),
                           original_inputs=inputs["input_hashes"],
                           package_versions={name: version(name) for name in ["numpy", "scipy", "pandas", "flowkit"]})
        mapped = directory / f"mapped_{fingerprint['implementation'][:12]}"
        saved_manifest = mapped / "manifest.json"
        if saved_manifest.exists():
            saved = json.loads(saved_manifest.read_text())
            if saved["fingerprint"] != fingerprint:
                raise ValueError("Cached Citrus cell mapping has stale inputs.")
            for name, expected in saved["outputs"].items():
                if digest(mapped / name) != expected:
                    raise ValueError("Cached Citrus cell mapping checksum mismatch.")
            visits.extend(pd.read_csv(mapped / "visits.csv", float_precision="round_trip").to_dict("records"))
            cell_records.append(pd.read_csv(mapped / "cells.csv"))
            audit.append(saved["audit"])
            print(f"Reuse verified Citrus full-donor mapping: split {sid}", flush=True)
            continue
        first_visit, first_record = len(visits), len(cell_records)
        tree = cKDTree(reference) if reference is not None else None
        positive_tree = cKDTree(reference[positive]) if tree is not None and positive.any() else None
        negative_tree = cKDTree(reference[~positive]) if tree is not None and not positive.all() else None
        native = pd.read_csv(directory / "native_map_checks.csv") if tree is not None else pd.DataFrame()
        checked = 0
        for donor in manifest["test_donors"]:
            data = values[donor]
            selected = np.zeros(len(data), dtype=bool)
            # Bound working memory and give progress on large full-donor files.
            for start in range(0, len(data), 10000):
                if tree is None:
                    break
                end = min(start + 10000, len(data))
                selected[start:end] = exact_positive_membership(tree, reference, positive, data[start:end], positive_tree, negative_tree, workers)
                check = native.loc[native.donor_id.eq(donor) & native.event_index.ge(start) & native.event_index.lt(end)]
                if len(check):
                    nearest = exact_nearest(tree, reference, data[check.event_index.to_numpy()], workers)
                    np.testing.assert_array_equal(nearest, check.nearest_train_index)
                np.testing.assert_array_equal(selected[check.event_index.to_numpy()], check.positive)
                checked += len(check)
            count = int(selected.sum())
            profile = data[selected].mean(axis=0, dtype=np.float64) if count else np.full(len(markers), np.nan)
            label = int(split.loc[split.donor_id.eq(donor), "label"].iloc[0])
            visits.append(dict(method="Citrus", split_id=sid, donor_id=donor, y_true=label,
                               n_cells=len(data), n_selected=count, selected_fraction=count / len(data),
                               **dict(zip(markers, profile))))
            map_cells = cells.loc[cells.sample_id.eq(donor)]
            cell_records.append(pd.DataFrame(dict(split_id=sid, cell_id=map_cells.cell_id,
                                                   positive=selected[map_cells.event_index.to_numpy()], negative=False)))
            print(f"Citrus split {sid}, {donor}: {count}/{len(data)} selected; native checks agree", flush=True)
        if tree is not None and checked != 384:
            raise ValueError("Incomplete native Citrus mapping audit.")
        audit.append(dict(split_id=sid, positive_clusters=len(expected_positive),
                           native_queries_checked=checked, native_index_mismatches=0,
                           max_centroid_error=manifest["max_centroid_error"],
                           null_positive_selection=tree is None))
        mapped.mkdir(exist_ok=True)
        pd.DataFrame(visits[first_visit:]).to_csv(mapped / "visits.csv", index=False)
        pd.concat(cell_records[first_record:], ignore_index=True).to_csv(mapped / "cells.csv", index=False)
        saved_manifest.write_text(json.dumps(dict(fingerprint=fingerprint, audit=audit[-1],
                                                  outputs={name: digest(mapped / name) for name in ["visits.csv", "cells.csv"]}), indent=2) + "\n")
        print(f"Citrus full held-out donors mapped: {sid + 1}/30 splits; {checked} native checks", flush=True)
    frequencies = aggregate_svm_frequencies(pd.concat(cell_records, ignore_index=True), cells, artifacts["splits"])
    frequencies = frequencies.drop(columns=["negative_count", "negative_frequency"]).assign(method="Citrus")
    return pd.DataFrame(visits), frequencies, pd.DataFrame(audit), source_hashes
