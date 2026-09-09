# Task 2: English slides and reports

The entry points are `slides.tex`, `short_report.tex`, `appendix.tex` and `detailed_report.tex`. Their checked PDFs are included alongside the sources. The talk has **8 main slides and 10 reserve slides**, targeting **7–10 minutes**. `speaker_notes.md` supplies a nine-minute English script and answers to likely questions.

The slides directly load **`../theme.tex`**, the existing Task 3 theme under `praesentation/`. Only the Task 2 footline is overridden locally. The repository-root files `slides/theme.tex` and `slides/main.tex` were empty in source commit `252415a`; they are not used or modified.

## Build the documents

From this directory, using pdfLaTeX and latexmk:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build slides.tex short_report.tex appendix.tex detailed_report.tex
```

The results appear under `build/`, which is ignored by Git. After checking a build, copy the four PDFs beside their `.tex` sources when preparing a version for sharing. Keep the `.tex` files, `figures/`, `data/`, `report_style.tex` and the parent `theme.tex` together. No Python execution or raw FCS files are required to read or compile the documents. Required LaTeX packages include Beamer/PGF, Latin Modern, English babel, booktabs, AMS math, graphicx, xcolor, hyperref and microtype.

For Overleaf, use the contents of this directory as the project root, add the existing parent `theme.tex` alongside them, and change the slide's first `\input{../theme}` to `\input{theme}`. Select the desired entry point as the main document. The report entry points already use files within this directory.

## Integrate the one-page report

`short_section.tex` has no document class or document wrapper. The standalone short report is exactly one page with 10-point type and 20 mm A4 margins. Its length may change under a different group-report template. The two-page appendix and the longer explanation are separate documents and are not automatically added to the five-page group report.

From a report whose working directory is the repository root, with the usual report preamble (including graphicx, booktabs and amsmath) loaded:

```latex
\newcommand{\TaskTwoRoot}{praesentation/aufgabe_02}
\input{praesentation/aufgabe_02/short_section}
```

Adjust `\TaskTwoRoot` if compiling from another directory. The section provides local fallback formatting for its table and captions; the standalone version uses the coloured definitions in `report_style.tex`. The rest of the group's existing report is not changed by this delivery.

## Small correction and asset export

Use the existing Python 3.12 project environment. From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m src.export_task2
```

The exporter reads the committed raw tables and saved embeddings. It never fits t-SNE or UMAP. It computes each neighbour index matrix once, with one thread and a 64 MiB working-memory setting; it reuses those indices for all comparisons. The overall process also holds input and output arrays, so 64 MiB is not a total-process RAM cap. During the concurrent Citrus run the command was also executed with `nice -n 10`.

`--update-notebook` additionally applies the documented partial notebook correction. It is not needed when merely rebuilding the presentation assets. The original raw tables, FCS files, ZIPs and paper PDFs are never rewritten. The notebook root finder now recognises the exported data instead of requiring the raw FCS directory, since Task 2 loads Parquet files.

### Corrections

The original `kneighbors()` call already excluded self; its extra `[:, 1:]` retained only 14 neighbours while dividing by 15. The corrected helper is in `src/task2_metrics.py`. A small independent identity test went from 14/15 to 1. Tests also check k=1, duplicate points, exact neighbour count, symmetry, geometric invariance, known overlap and invalid inputs.

The final quality reference is explicitly the **29 selected PCs**. The current saved Pearson distance correlations match the committed 29-PC array; the original quality-code variable named the initially fitted 30-PC array. All final kNN values were recalculated against the explicit 29-PC reference. The all-pairs Procrustes results were reproduced, and Pearson distance correlations were checked using the same seeded 2,000-cell sample.

The final tables and pairwise heatmap in the notebook have been updated. In the stored sweep tables, invalid kNN columns were removed, but the Trustworthiness columns and the original plot outputs were retained. A future complete notebook run will compute the sweep kNN columns with the corrected helper. Updated cells have no claimed sequential execution count; this delivery is a partial correction, not a claim of a fresh top-to-bottom notebook run.

### What has not been rerun

The t-SNE and UMAP embeddings, the six t-SNE and sixteen UMAP sweeps, the 40,000-cell Trustworthiness and silhouette calculations, and the GMM sampling check were not rerun. The t-SNE selection still uses the highest saved Trustworthiness, at perplexity 60; its advantage over 50 is only 0.000073. We do not claim a corrected sweep-wide kNN winner or a robust optimum across seeds.

### Provenance and interpretation

`data/provenance.json` records source commit `252415a`, the corrected notebook hash, input hashes, sample sizes, seed, reference dimension and the original source cells for retained PNG images. CSV and LaTeX tables are generated together; documents share these assets. PCA variance curves and final-map plots are drawn from the existing measurements or coordinates. The arcsinh and ideal Shepard plots are explicitly mathematical illustrations. Selected parameter panels are cropped and rearranged from the original raster plots; no coordinates are altered or invented.

For readability, the parameter panels' small raster titles are replaced by larger vector labels. The plotted pixels are retained; the eight-panel reserve views also show the original rounded Trustworthiness values. The exported software versions are recorded in the provenance file.

Important distinctions are stated in all documents: 40,000 analysis cells versus 261,593 available gated NK cells; 29 PCs retained versus only two displayed; local-neighbour overlap versus Procrustes shape; donor CMV labels versus infected-cell labels. Full-sample cell averages weight the 20 donors equally because each contributes 2,000 cells; the 5,000-cell sweep has 220–281 cells per donor and is only approximately balanced.

## Verification performed

Five deterministic unit tests passed. The exporter successfully checked row alignment, equal donor counts, exactly 15 non-self neighbours, PCA coordinate agreement, the selected parameter settings and the saved Procrustes/Pearson values. Notebook schema and code syntax, the exported-data root lookup, the helper imported by the notebook, input hashes and removal of invalid sweep columns were checked separately.

All PDFs compiled without overfull/underfull box warnings: **18 slide pages, 1 short-report page, 2 appendix pages and 14 detailed-report pages**. Every page was rendered and visually inspected; PDF text boundaries were also checked. The short section compiled successfully inside a separate article with a standard preamble and an explicit `\TaskTwoRoot` path. No fresh top-to-bottom notebook execution or new embedding/classification benchmark is claimed.

## Sharing through Git

Include the Task 2 sources, checked PDFs, figures, data tables, speaker notes, new Python helpers/tests and corrected notebook in the eventual commit. Build intermediates stay ignored. No new raw input data need to be added. The existing theme is already versioned. Changes were prepared on `slides-aufgabe-2` in the separate worktree; no commit, merge or push is performed by the export/build commands.
