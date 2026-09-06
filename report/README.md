# Editing the report

Open `main.tex` in VS Code. LaTeX Workshop builds on save and shows the PDF in a
tab beside the source. Use **LaTeX Workshop: View LaTeX PDF file** in the command
palette to open the preview. SyncTeX connects source locations and PDF positions.
The output is `report/build/main.pdf`.

From the repository root, the same build is:

```bash
latexmk -cd -pdf -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=build report/main.tex
```

TeX Live with `latexmk`, pdfLaTeX, Latin Modern, `natbib`, `booktabs`, `geometry`,
`caption` and `microtype` is sufficient. In this WSL environment, TinyTeX (a small
TeX Live distribution) is installed in `~/.TinyTeX`, with commands linked into
`~/.local/bin`; LaTeX Workshop is installed on the WSL side. A fresh machine needs
these tools installed separately. If VS Code was already running during installation
and cannot find `latexmk`, reload the window and ensure `~/.local/bin` is on PATH.
See the [TinyTeX installation guide](https://yihui.org/tinytex/) and
[LaTeX Workshop documentation](https://github.com/James-Yu/LaTeX-Workshop/wiki).

## Content and figures

- Edit prose and captions in `main.tex`, and references in `references.bib`.
- Replace the author placeholder before submission.
- The Task 6 minipage reserves half the text height of one page. Replace the whole
  minipage when the extension is ready. No extension results are implied by it.
- Explicit page breaks organize the initial five-page draft. Recheck page count
  after larger edits; the five-page limit includes figures, tables and references.
- The selected PDFs in `figures/` and LaTeX table in `tables/` belong to the report.
  They make normal compilation independent of the local dataset, Python and R.

To refresh those assets after an intentional analysis update, activate the project's
Python environment and run from the repository root:

```bash
python -m src.report_assets
```

The exporter reads saved results under `results/tables/`. Figure 1 also reads the
exact saved events from the original `gated_alive` FCS files for CD3 colours and
checks their exploration checksum. It does not fit projections, clustering or
classifiers, and does not change the analysis outputs. Missing or inconsistent
inputs must be resolved in the analysis before exporting. Run this command
separately; normal LaTeX builds never run analyses or regenerate figures.

Figure sources: `task2_embeddings`, `task2_pairwise_jaccard` and `task2_cells`
(Figure 1); `task3_marker_profiles`, `task3_recommendations` and saved exploratory
scaling (Figure 2); `task5_cell_scores` and `task5_report_marker_table` (Figure 3).
The classification table uses `task4_metric_summary`, checked against
`task4_split_metrics`. All these inputs are CSV/JSON exports in `results/tables/`.
The first report draft uses the complete `gated_alive` runs from 5 September 2026.

Generated assets do not automatically rewrite claims in the prose: check numbers
and conclusions in the text whenever the underlying analyses change. Build files
under `build/` are ignored by Git; source, selected assets and project VS Code
settings are intended to be versioned. No raw data are included.
