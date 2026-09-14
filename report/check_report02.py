"""Check V2 document constraints and numerical/preservation evidence separately.

Use PYTHONPATH=/tmp/report_korrigiert_pdf_tools with the project Python if needed.
This script writes only report/data/v02/validation.json.
"""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
DATA = REPORT / "data/v02"
ASSETS = ROOT / "praesentation/report_assets"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


checks = {}
protected = json.loads((DATA / "protected_v1.json").read_text())
originals = json.loads((REPORT / "data/protected_files.json").read_text())
for name, expected in (protected | originals).items():
    assert sha(ROOT / name) == expected, f"Protected file changed: {name}"
checks["protected_v1_files_unchanged"] = len(protected)
checks["colleague_and_presentation_files_unchanged"] = len(originals)

manifest = json.loads((DATA / "figure_provenance.json").read_text())
assert sha(REPORT / "export_assets02.py") == manifest["renderer_sha256"]
for name, expected in manifest["inputs_sha256"].items():
    assert sha(ROOT / name) == expected, name
for name, expected in manifest["figures_sha256"].items():
    assert sha(REPORT / name) == expected, name
checks["figure_input_and_output_hashes_match"] = True

sources = {name: (REPORT / f"{name}.tex").read_text() for name in ["main", "main02", "supplement", "supplement02"]}
assert sources["main02"].startswith(r"\documentclass[11pt,a4paper]{article}")
assert sources["supplement02"].startswith(r"\documentclass[11pt,a4paper]{article}")
assert len(re.findall(r"\\includegraphics", sources["main02"])) == 4
assert "{clustering.pdf}" in sources["main02"]
checks["main_figures"] = 4
checks["body_format"] = "A4, 11pt, unchanged shared style"

math = lambda s: re.findall(r"\$.*?\$|\\\[.*?\\\]", s, re.S)
assert math(sources["supplement"]) == math(sources["supplement02"])
assert re.findall(r"\d+(?:\.\d+)?", sources["supplement"]) == re.findall(r"\d+(?:\.\d+)?", sources["supplement02"])
checks["supplement_math_and_numeric_tokens_unchanged"] = True
checks["semicolon_counts_tex"] = {name: s.count(";") for name, s in sources.items()}
assert sources["main02"].count(";") < sources["main"].count(";")
assert sources["supplement02"].count(";") < sources["supplement"].count(";")

documents = {name: pymupdf.open(REPORT / f"build/{name}.pdf") for name in ["main02", "supplement02"]}
assert len(documents["main02"]) == 5
checks["main_pages"] = len(documents["main02"])
checks["supplement_pages"] = len(documents["supplement02"])
text = {name: "\n".join(p.get_text() for p in doc) for name, doc in documents.items()}
for name, doc in documents.items():
    assert "??" not in text[name]
    assert not re.search(r"\[\s*\?", text[name])
    log = (REPORT / f"build/{name}.log").read_text(errors="replace")
    for warning in ["Overfull", "Float too large", "undefined citations", "undefined references", "Citation(s) may have changed"]:
        assert warning not in log, (name, warning)
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    assert page.rect.contains(pymupdf.Rect(span["bbox"])), (name, page.number, span["text"])
checks["both_builds_without_overflow_or_unresolved_references"] = True
for title in ["Introduction", "Methods", "Results", "Discussion and conclusion", "References"]:
    assert title in text["main02"]
for author in ["Sebastian Fay", "Marie Schygulla", "Gregor Habitzreither"]:
    assert author in text["main02"]
for n in range(1, 6):
    assert f"S{n} " in text["supplement02"]

bonus = []
extent = []
for page in documents["main02"]:
    top, bottom = 1.8 * 72 / 2.54, page.rect.height - 2.5 * 72 / 2.54
    blocks = [b for b in page.get_text("blocks") if b[4].strip() != str(page.number + 1)]
    extent.append(round((max(b[3] for b in blocks) - top) / (bottom - top), 4))
    start, end = page.search_for("Exploratory extension:"), page.search_for("Discussion and conclusion")
    if start:
        assert end, "Bonus and following discussion must share a page."
        block_text = page.get_text(clip=pymupdf.Rect(0, start[0].y0 - 2, page.rect.width, end[0].y0))
        for value in ["0.8075", "0.9125", "0.7375"]:
            assert value in block_text, "Keep the bonus table inside the bonus block."
        height = end[0].y0 - start[0].y0
        assert height / (bottom - top) <= .5
        bonus.append(dict(page=page.number + 1, height_points=height,
                          fraction_of_text_height=height / (bottom - top)))
assert len(bonus) == 1
assert min(extent) > .9, "Check for a sparsely filled main page."
checks["bonus_extent"] = bonus[0]
checks["main_content_bottom_as_fraction_of_text_height"] = extent

footers = {"dimensionality_reduction": ["Pairwise Procrustes disparity"],
           "classification": ["30 shared splits", "Paired counts:"],
           "subsets": ["30 shared splits", "Profiles: selected"]}
figure_heights = {}
for name, forbidden in footers.items():
    old = pymupdf.open(REPORT / f"figures/{name}.pdf")
    new = pymupdf.open(REPORT / f"figures/v02/{name}.pdf")
    fig_text = new[0].get_text()
    assert not any(s in fig_text for s in forbidden)
    assert new[0].rect.height < old[0].rect.height
    figure_heights[name] = dict(v1=old[0].rect.height, v2=new[0].rect.height)
checks["figure_heights_points"] = figure_heights
checks["prose_footers_removed_and_canvas_reduced"] = True

# Added numeric claims are checked against the existing result exports.
pairs = read(ASSETS / "aufgabe_02/data/pairwise.csv")
for value in pairs.procrustes_disparity:
    assert f"{value:.3f}" in sources["main02"]
tsne = read(ASSETS / "aufgabe_02/data/tsne_sweep.csv").set_index("perplexity")
for perplexity in [5, 30, 70]:
    assert f"{tsne.loc[perplexity, 'trustworthiness']:.4f}" in sources["main02"]
umap = read(ASSETS / "aufgabe_02/data/umap_sweep.csv")
lr_column = "learning_rate"
nn_column = "n_neighbors"
fixed = umap.loc[(umap[nn_column].eq(5) & umap.min_dist.eq(0)) |
                 (umap[nn_column].eq(50) & umap.min_dist.eq(.1))]
for rate in [.1, 5]:
    value = fixed.loc[fixed[lr_column].eq(rate), "trustworthiness"].mean()
    assert f"{value:.4f}" in sources["main02"]
profiles = read(REPORT / "data/cluster_profiles.csv")
leiden = profiles.loc[profiles.method.eq("Leiden") & profiles.cluster.eq(16)]
assert leiden.cells.eq(281).all()
assert (leiden.loc[leiden.marker.isin(["NKG2C", "CD57"]), "relative_iqr"] > 2).all()
provenance = json.loads((ASSETS / "aufgaben_04_05/data/provenance.json").read_text())
assert [provenance["results"][m]["selected_every_test_visit"] for m in ["CellCNN", "SVM", "Citrus"]] == [15, 0, 0]
bonus_data = read(ASSETS / "aufgabe_06/data/summary.csv")
for value in bonus_data.network_auc_mean:
    assert f"{value:.4f}" in sources["main02"]
assert bonus_data.network_splits.eq(100).all()
assert bonus_data.frequency_splits_common.eq(98).all()
checks["added_numeric_claims_match_sources"] = True
checks["pdf_sha256"] = {name: sha(REPORT / f"build/{name}.pdf") for name in documents}
(DATA / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks, indent=2))
