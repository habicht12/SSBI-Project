# Task 6: CellCNN model comparison

[slides.pdf](slides.pdf) contains five main slides and six backup slides in the
existing 16:9 Beamer theme. [slides.tex](slides.tex) is the editable source.
Plots, tables, captions and metadata are in English. Comments next to substantive
LaTeX edits explain the methodological or presentation reason for each change.

## Regenerate plots and build slides

From the repository root, in the existing Python project environment:

```bash
python -m src.export_task6
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir=build praesentation/aufgabe_06_vergleich/slides.tex
cp praesentation/aufgabe_06_vergleich/build/slides.pdf praesentation/aufgabe_06_vergleich/slides.pdf
```

The exporter uses only NumPy, pandas, Matplotlib and the CSV files in `data/`.
It does not execute notebooks, train models or read FCS files. No GPU is needed.
The existing shared [theme](../theme.tex) is retained without changes.

## Data and provenance

The data are unchanged copies of the
[audited 100-split benchmark](https://github.com/habicht12/SSBI-Project/tree/1f00db23b0a2fc4a4052d113285feef478fc69c4/results/tables/task6_comparison_100)
from `GrHa-learnable-pooling`. Only the plotting code and necessary result tables
were brought to `main`; no training notebooks or model checkpoints were imported.

- `input_provenance.json`: source commit and hashes of the six input CSV files.
- `summary.csv`, `split_metrics.csv`: reported summary and per-split metrics.
- `mean_rocs.csv`: stored mean interpolated ROC curves and original mean split AUCs.
- `frequencies.csv`: donor/test-visit frequencies, counts and half-max thresholds.
- `paired_differences.csv`: within-split AUC differences relative to CellCNN.
- `learned_alphas.csv`: filter parameters from the selected models.
- `provenance.json`: English export rules, generated claims and output hashes.

Input hashes and numerical consistency are checked before rendering. Display
labels are translated in the exporter; the source CSV bytes are not edited.

## What the plots show

- Network metrics use 100 common outer splits. Each test set has two CMV-positive
  and four CMV-negative donors. Each donor score averages five bags of 20,000 cells.
- Frequency comparisons use the same 98 jointly evaluable splits. CellCNN has no
  positive-output filter in splits 44 and 84. Missing values are not set to zero.
- Half-max membership is strictly `h > 0.5 * max(inner-training responses)` for the
  largest positive output-contrast filter, including when the training maximum is
  zero. The evaluation sample contains 20,000 fixed cells per test donor.
- `frequency_auc.pdf` and `frequency_effect.pdf` replace the combined
  `frequency_metrics.pdf`. Only the frequency-effect plot is used on the main slide.
  The effect is mean CMV-positive frequency minus mean CMV-negative frequency,
  multiplied by 100 to give percentage points; it is not a standardized effect size.
- Boxplots show Q1–Q3, median and individual split values, not confidence intervals.
- Donor plots use one point per donor after averaging available test visits in the
  common splits. All panels share the same y-axis.
- Alpha plots show 416 Quadratic and 414 Prototype filter parameters across
  100 selected models per variant. Alpha is a pooling fraction, not population frequency.

There are 20 independent donors (9 CMV-positive, 11 CMV-negative), not 600 independent
observations per method. The selected inner model uses 9–10 training donors; there
is no refit on all 14 outer-training donors. The comparison remains descriptive.
CMV is a donor-status label, not a cell-type annotation; half-max selection is not clustering.

## Verified result

Both quadratic variants have mean network AUC **0.9125**. Soft versus hard pooling
is better in 5 splits, tied in 90 and worse in 5: the mean AUC difference is zero.
The largest mean frequency difference is CellCNN's **0.740 percentage points**;
this does not establish a superior biological phenotype.

The final PDF is checked by rendering every page, inspecting plots and equations,
and reviewing the LaTeX log for layout warnings. Targeted regression tests run with:

```bash
python -m unittest discover -s tests -p test_export_task6.py -v
```
