"""Validate document constraints, source preservation and report result exports.

Run with a PyMuPDF installation on PYTHONPATH if it is not in the project env.
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
read = lambda p: pd.read_csv(p, float_precision="round_trip")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks = {}

protected = json.loads((REPORT / "data/protected_files.json").read_text())
assert all(sha(ROOT / name) == expected for name,expected in protected.items())
assert not list((REPORT / "build").glob("detailbericht_de.*"))
checks["protected_originals_unchanged"] = list(protected)
checks["legacy_detail_report_removed"] = True

main = pymupdf.open(REPORT / "build/main.pdf")
supplement = pymupdf.open(REPORT / "build/supplement.pdf")
assert len(main) == 5, f"Main report has {len(main)} pages"
main_text = "\n".join(page.get_text() for page in main)
sup_text = "\n".join(page.get_text() for page in supplement)
assert "??" not in main_text + sup_text
assert not re.search(r"\[\s*\?", main_text + sup_text)
for name in ["Sebastian Fay", "Marie Schygulla", "Gregor Habitzreither"]:
    assert name in main_text
for title in ["Introduction", "Methods", "Results", "Discussion and conclusion", "References"]:
    assert title in main_text
for n in range(1, 6):
    assert f"S{n} " in sup_text
assert len(re.findall(r"\\includegraphics", (REPORT / "main.tex").read_text())) == 4
checks["main_pages"] = len(main)
checks["supplement_pages"] = len(supplement)
checks["main_figures"] = 4

locations = []
for i,page in enumerate(main):
    start = page.search_for("Exploratory extension:")
    end = page.search_for("Discussion and conclusion")
    if start:
        assert end, "Bonus and discussion are split across pages"
        height = end[0].y0 - start[0].y0
        body_height = page.rect.height - (1.8+2.5)*72/2.54
        assert .30 <= height/body_height <= .5, (height, body_height)
        locations.append(dict(page=i+1, height_points=height, fraction_of_text_height=height/body_height))
assert len(locations) == 1
checks["bonus_extent"] = locations[0]

for document in ["main", "supplement"]:
    log = (REPORT / "build" / f"{document}.log").read_text(errors="replace")
    for text in ["Overfull", "Float too large", "undefined citations", "undefined references", "Citation(s) may have changed"]:
        assert text not in log, (document,text)
    checks[f"{document}_build_without_overflow_or_unresolved_references"] = True
for page in main:
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:continue
        for line in block["lines"]:
            for span in line["spans"]:
                assert page.rect.contains(pymupdf.Rect(span["bbox"])), span["text"]
checks["main_text_inside_page_bounds"] = True

for file,key,base in [("figure_provenance.json","figures_sha256",REPORT),
                      ("table_provenance.json","table_sha256",REPORT / "tables")]:
    provenance = json.loads((REPORT / "data" / file).read_text())
    for name,expected in provenance[key].items():assert sha(base / name) == expected, name
checks["all_figure_and_table_checksums_match"] = True

base = ROOT / "praesentation/report_assets"
metric = read(base / "aufgaben_04_05/data/metric_summary.csv")
for method,value in [("CellCNN",.875),("SVM",.875),("Citrus",.625)]:
    assert metric.loc[metric.method.eq(method)&metric.metric.eq("roc_auc"),"median"].iloc[0] == value
bonus = read(base / "aufgabe_06/data/summary.csv").set_index("model")
for name,value in [("cellcnn",.8075),("quadratic_top1",.9125),("quadratic_soft",.9125),("prototype_soft",.7375)]:
    np.testing.assert_allclose(bonus.loc[name,"network_auc_mean"],value,atol=1e-12,rtol=0)
assert bonus.network_splits.eq(100).all() and bonus.frequency_splits_common.eq(98).all()
clusters = read(REPORT / "data/cluster_summary.csv")
assert clusters.clusters.tolist() == [4,6,18]
np.testing.assert_allclose(clusters.db.round(3),[2.363,2.522,2.361],atol=1e-12)
checks["key_results_and_analysis_scopes_match"] = True
checks["pdf_sha256"] = {name:sha(REPORT / "build" / name) for name in ["main.pdf","supplement.pdf"]}
(REPORT / "data/validation.json").write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
