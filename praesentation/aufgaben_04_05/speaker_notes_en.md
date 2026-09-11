# Speaker notes for Tasks 4 and 5

English speaking script for the 13 main slides, planned for **10:00 minutes**.
The seven backup slides are outside the regular talk. These timings are targets,
not a measured presentation duration. Read the quoted passages as the script;
the additional notes are delivery cues or material for questions.

| Slide | Duration | Finish at |
|---|---:|---:|
| 1 – Data and evaluation setup | 0:35 | 0:35 |
| 2 – CellCNN | 0:55 | 1:30 |
| 3 – Setting up Citrus | 0:30 | 2:00 |
| 4 – Citrus workflow | 0:45 | 2:45 |
| 5 – SVM cell score | 0:40 | 3:25 |
| 6 – SVM donor score | 0:55 | 4:20 |
| 7 – Performance comparison | 0:50 | 5:10 |
| 8 – CellCNN cell selection | 0:35 | 5:45 |
| 9 – Citrus cluster selection | 0:35 | 6:20 |
| 10 – SVM cell selection | 0:45 | 7:05 |
| 11 – Recurring subsets | 1:00 | 8:05 |
| 12 – Selected cells | 0:55 | 9:00 |
| 13 – Biological interpretation | 1:00 | 10:00 |

## 1. From cells to donor-level prediction — 35 seconds

> Task 4 predicts a donor's CMV status. We have 20 donors and 37 markers.
> Each marker value receives an ArcSinh transformation with cofactor five.
> CellCNN and SVM also use scaling fitted on training donors. Each split assigns
> 14 donors to training and six to testing; three inner folds select the model.
> All methods share the same 30 splits. Cells belonging to one donor always stay
> together throughout this process.

Point to the split diagram. Donors are the independent observations.
`gated_alive` contains live, singlet PBMCs without an additional NK gate.
In the formula, i identifies the cell and j the marker; x is the raw measurement
and a its transformed value.

## 2. CellCNN learns filters for informative cells — 55 seconds

> CellCNN receives groups of 3,000 cells. Each filter computes a weighted sum of
> standardized markers plus a bias. ReLU sets negative responses to zero.
> We then average the highest one percent of responses per filter, allowing a
> small population to contribute to the CMV probability. Filters and prediction
> are trained jointly from donor labels. We generate 200 cell groups per training
> donor and select the best network through inner validation. Our PyTorch
> implementation follows the paper and reference code, with a fixed search over
> three, four, or five filters. At test time, we average five input probabilities.
> We do not refit the selected network after model selection.

Follow the diagram. The legend defines cell i, filter f, marker vector z, weights w,
bias b, and response r. Each test input contains up to 20,000 cells. The fixed
filter search replaces the paper's random search. Half-maximum selection on
slide 8 serves a different purpose from pooling.

## 3. Running the original Citrus implementation — 30 seconds

> For Citrus, we run the original package code in a separate Conda environment
> with R 4.5. It also provides the libraries and C++ compiler. Citrus 0.8 and
> Rclusterpp are installed at fixed Git commits. A dedicated R Jupyter kernel
> runs the notebook with our shared donor folds. CellCNN was reimplemented
> from the paper; Citrus uses its original implementation. Both routes follow
> the paper methodologically.

“Legacy” refers to the Citrus package. The recorded run uses R 4.5.3 and
Rclusterpp 0.2.6. `flowCore` reads FCS files; `glmnet` supplies regularized
regression. Setup is documented in the project README and `environment-citrus.yml`.
Do not imply additional source-code repairs. The pinned commits fix Citrus and
Rclusterpp sources, not every transitive dependency version.

## 4. Citrus predicts from population frequencies — 45 seconds

> Citrus starts with equal numbers of cells per training donor and builds a
> hierarchical Ward tree. It retains clusters containing at least 0.05 percent
> of training cells; parent and child clusters may overlap. Each donor is then
> represented by its cell frequency in these clusters. New cells inherit the
> memberships of their nearest training cell. Finally, L1-logistic regression
> selects predictive frequencies and returns a CMV probability. Population
> discovery precedes supervised classification. Our 10,000 cells per donor
> and 30 splits are a computational compromise compared with 20,000 cells
> and 100 splits in the paper.

Explain the three steps on the left while following the vertical flow on the right.
Clusters are not mutually exclusive cell types. Inner trees use their respective
inner training donors. The original lambda-grid limitation is covered on backup 14.

## 5. A linear SVM assigns a score to each cell — 40 seconds

> The linear single-cell SVM receives 10,000 training cells per donor. Each cell
> inherits its donor's CMV label, which is weak supervision. The SVM computes
> a weighted sum of standardized markers plus an intercept. This is the cell
> score, also called the margin. Higher values point more strongly in the
> positive model direction; they are not probabilities. Training uses L2
> regularization and squared-hinge loss. We then need to turn these cell scores
> into one score per donor.

The drawing illustrates two markers; the model uses all 37. Point colours denote
donor labels, not validated cell states. The margin is a functional decision score,
not a geometric distance divided by the weight norm. The training objective is
explained on backup 15.

## 6. Our adaptation: aggregate the highest cell margins — 55 seconds

> We average the highest one percent of cell margins within a donor, rounding
> the cell count up. For a thousand cells, we keep ten. The example margins
> from 2.1 to 3.0 produce a donor score of 2.55. At an illustrative threshold
> of 1.8, the prediction is positive. In the benchmark, we choose loss weight C
> by mean inner donor AUC. The threshold comes from inner out-of-fold donor
> scores, maximizing sensitivity plus specificity. We then refit the scaler
> and SVM on all 14 training donors. This top-one-percent aggregation is our
> project adaptation, fixed before testing.

The legend defines donor d, cell count N, selected count k, index set I, cell
margin m, donor score s, and threshold tau. C weights cell losses against
regularization; it is not a learning rate. The numbers and diagram widths are
illustrative. AUC uses continuous donor scores. Outer test donors choose neither C
nor the threshold. See backup 16 for the complete selection and test workflow.

## 7. Performance across 30 shared donor splits — 50 seconds

> Here we compare all three methods on the same 30 splits. Each dot shows
> ROC-AUC on six test donors; each box spans the first to third quartiles.
> CellCNN and SVM reach a median AUC of 0.875, and Citrus 0.625. Variation is
> substantial. The table adds average precision and balanced accuracy; CellCNN
> and SVM share these medians too, although individual predictions can differ.
> Balanced accuracy also depends on the decision threshold. Quartiles describe
> variation across splits. Since splits overlap and reuse the same 20 donors,
> these results do not establish general superiority.

The plot explanation now sits directly below the boxplots. BA thresholds are
0.5 for CellCNN/Citrus and inner-trained for SVM. Two positive and four negative
test donors yield only eight positive–negative pairs, explaining the coarse AUC
steps. Tied scores contribute half a correctly ranked pair. Quartiles are not
confidence intervals.

## 8. CellCNN: select cells with a strong filter response — 35 seconds

> Task 5 now asks which cells are selected. For CellCNN, we divide a filter's
> response by its maximum on the original training reference. A cell passes
> when this normalized score is strictly above 0.5. If the maximum is ten,
> responses of eight and six pass, but five does not. This half-maximum rule
> defines the interpreted subset. Top-one-percent pooling in Task 4 produces
> the prediction instead.

Each score belongs to one cell and one filter. Direction comes from the difference
between the two output weights, not the nonnegative ReLU response. Interpret only
filters with a positive reference maximum and nonzero output contrast. The reference
uses the nine or ten actual inner training donors of the selected network.
See backup 18 for details.

## 9. Citrus: select clusters used by the classifier — 35 seconds

> Citrus selects whole clusters through their role in the classifier. Our
> quantitative selection score is the absolute regression coefficient of a
> cluster-frequency feature. Values above a small numerical-zero tolerance
> are retained. The sign gives the direction: a positive coefficient means
> that increasing this frequency shifts the prediction towards CMV-positive,
> holding other features fixed. We use the saved mean marker profiles of
> these clusters, so our Citrus map displays centroids.

The tolerance is ten to the minus ten. It defines numerical zero, not biological
relevance or statistical significance. All effective clusters are included. The
exports do not contain the full tree and its memberships; exact Citrus selection
of individual map cells cannot be recovered from them. See backup 19.

## 10. SVM: select the strongest positive cell scores — 45 seconds

> For the SVM, we first calculate the margin of every cell in a full test donor.
> We retain the highest one percent within that donor. Positive selection also
> requires exceeding the learned donor threshold. Subtracting the threshold
> from the cell margin gives a centered score; a positive result satisfies this
> second condition. Both conditions must hold together. Each model evaluates
> only its outer test donors. We then restrict the display to cells present
> on the existing map.

The threshold comes from inner validation. It is neither a newly fitted cell
threshold nor automatically zero. Selection uses the complete sample, not the
500 displayed cells per donor. Positive selection describes model direction;
it does not demonstrate that a cell is infected with CMV. Backup 20 explains
boundary ties, negative selection, and selection frequencies.

## 11. Which CellCNN and Citrus subsets recur? — 60 seconds

> Once subsets are defined, we compare their mean marker profiles across splits.
> Similar centroids are grouped separately for CellCNN and Citrus. We count
> distinct splits containing each group; multiple members within one split
> count only once. Groups occurring in at least six of 30 splits appear on
> this t-SNE map. The most frequent CellCNN group appears in all 30 splits;
> the most frequent Citrus group appears in 18. Both are positively associated
> with CMV. Stars mark saved representatives. These points are projected
> training centroids; matching colours across panels do not identify the
> same population.

The map contains **10,000 `gated_alive` cells**, 500 per donor, at perplexity 30.
It differs from the 40,000-NK-cell map in the Task 2 slides on `main`. Centroids
are projected through the nearest map cell. The map and common profile scaling
were established exploratively using all donors; neither enters classifier evaluation.

Grouping uses average linkage, cosine distance, and a cutoff of 0.4. Transferring
this rule from the reference code's filter grouping to subset centroids is a
documented adaptation. The three Citrus null models remain in the denominator.
Group IDs are method-specific. The recurrence cutoff is not a significance test.

## 12. Which cells does the SVM repeatedly select? — 55 seconds

> On the left, one representative CellCNN filter selects 87 map cells, applied
> exploratively to all donors. On the right, 93 map cells were selected
> positively by the SVM at least once. Colour shows the fraction of their
> donor's test appearances in which this happened. The denominator includes
> every test appearance, even without selection; each donor has four to 16.
> A cell selected positively three times in eight appearances therefore has
> frequency three eighths. This recurrence of the same cell differs from
> recurrence of similar centroid groups on the previous slide.

The denominators differ, so the frequencies are not comparable effect sizes.
The 93 cells are highlighted map cells, not the total selected population.
SVM selection uses the full test donor. For its cell map, CellCNN uses the
unchanged representative of the most frequent group; this is not an outer-test
evaluation of every displayed cell.

## 13. What do the marker profiles support? — 60 seconds

> The heatmap shows saved group representatives for CellCNN and Citrus. For
> the SVM, we average positively selected cells within each test visit, then
> visits within donors, and finally all 20 donors equally. There are 168
> nonempty visits out of 180. CellCNN shows elevated NKG2C and CD57 with
> NK-associated markers, compatible with a memory-like NK profile. CD3 and
> CD57 suggest a T-associated pattern for Citrus. The SVM also shows elevated
> NKG2C, CD57, and NK markers; CD3 lies above the reference mean. The shared
> z-scale supports comparison, but subset definitions differ. These profiles
> support biological hypotheses, not validated cell-type assignments.

The SVM profile uses all cells from the respective outer test donors in splits
0–29, regardless of true labels or predicted class. It is not restricted to the
93 map cells. Twelve test visits have no positive selection: their profiles are
missing, not zero. Nonempty visits receive equal weight within each donor, and
donors receive equal weight in the final profile. The profile describes expression
conditional on positive selection, not selection frequency.

Representatives remain CellCNN split 9/filter 2 and Citrus split 25/cluster 139891.
The SVM row is not a group representative. z-values are not positivity gates.
Mean profiles can summarize heterogeneous cells; CD3 is not zero in the CellCNN
representative either.

## 14. Backup: hyperparameters and paper deviations

CellCNN retains the selected inner network, trained on nine or ten donors. Citrus
and SVM use all 14 outer training donors in their final models. Citrus fits inner
trees per fold, but its original lambda grid uses cluster features and labels
from all 14 outer training donors. Outer test donors remain excluded from fitting.
Do not claim complete separation of every data-dependent inner decision.

CellCNN validation ties are resolved by AUC, loss, fewer filters, and fold order.
SVM ties favour smaller C. Citrus `cv.min` minimizes inner classification error,
favouring stronger regularization in a tie. The 37 supplied markers differ from
the 36 reported in the paper. Different cell samples and GPU/CPU use limit direct
runtime comparisons.

The average-linkage/cosine/cutoff-0.4 rule was transferred from the reference code's
filter grouping to subset centroids. It is not a confirmed identical reproduction
of the original NK-centroid grouping parameters.

## 15. Backup: SVM approach 1 — training the cell scorer

The left column covers preprocessing. Draw 10,000 cells per training donor without
replacement using fixed seeds. Apply ArcSinh with cofactor five, fit a shared
StandardScaler on these training cells, and apply it unchanged to validation or
test donors. Equal cell counts give donors equal weight in the loss; no extra
class weighting is applied. Cells inherit donor labels as weak supervision.

The right column shows the actual `LinearSVC` objective: L2 penalizes large
parameters, and squared-hinge loss penalizes violations of the desired label
margin. C weights cell losses relative to regularization. Mathematical label y
uses minus or plus one; t remains reserved for the centered interpretation score.
With `intercept_scaling=1`, the intercept is regularized too, explaining its
squared term in the formula. The SVM learns 37 weights and one intercept without
probability calibration.

Numerical settings: `dual="auto"`, at most 10,000 iterations, tolerance 0.0001.
With many more cells than markers, optimization is primal here. There is no
validation-based early stopping as in CellCNN.

## 16. Backup: SVM approach 2 — model selection and test

Within the 14 outer training donors, compare C = 0.01, 0.1, and 1 using three
fixed donor folds. Each fold uses identical cell samples and scalers across C
values. Highest mean inner donor AUC wins; ties favour smaller C. For the selected
C, collect 14 out-of-fold scores, one per training donor. Select the finite ROC
threshold maximizing the Youden index: sensitivity plus specificity minus one.
Ties choose the highest offered optimal threshold.

Then draw a new deterministic sample from all 14 training donors and refit the
scaler and final SVM. Keep the inner-selected threshold. For each of the six test
donors, score every cell, average the highest one percent, and compare with the
threshold. Equality gives a positive donor prediction, whereas positive individual
cell selection in Task 5 requires strict exceedance. AUC uses continuous scores;
BA uses binary predictions.

Inner-model scores are uncalibrated and may differ in scale from one another and
from the refitted model. Threshold transfer is therefore a limitation. Neither
model nor threshold selection uses outer test donors. Historical inner individual
scores are not recomputed; these slides explain the existing workflow.

## 17. Backup: CellCNN and SVM across 100 splits

This comparison includes only CellCNN and SVM. Both again reach median AUC 0.875.
Median balanced accuracy is 0.750 for CellCNN and 0.625 for SVM. The 600 test
predictions per method repeatedly evaluate the same 20 donors. Keep this pairwise
comparison separate from the three-method comparison over 30 splits; Citrus is
not part of this table.

## 18. Backup: CellCNN subset selection in detail

The left column explains response and reference. The selected network was trained
on nine or ten inner training donors. We reconstruct 20,000 reference cells per
donor with the original candidate seed, without refitting the scaler or network.
The nonnegative ReLU response is divided by the filter's reference maximum.
A response exactly at half-maximum is excluded.

On the right, direction follows from the difference between CMV-positive and
CMV-negative output weights. Filters with zero reference maximum or zero output
contrast provide no relevant subset. For effective filters, average the selected
reference cells' ArcSinh marker vectors to obtain 37 centroid values. Selected
cells receive equal weight; donors contributing more selected cells have greater
influence. Normalized scores are not probabilities and can exceed one on new cells.
This subset is not an exact list of cells pooled in every prediction.

## 19. Backup: Citrus subset selection in detail

The linear logistic predictor combines cluster-frequency features and coefficients.
`glmnet` standardizes frequency features internally. The ArcSinh-transformed markers
receive no additional z-scoring before clustering. All coefficients with absolute
value above ten to the minus ten are retained. Their signs describe the conditional
effects of the respective frequencies.

Each saved centroid averages the ArcSinh profiles of its original training-cluster
cells. Parent and child clusters can contain the same cells. Centroids and
coefficients alone cannot assign arbitrary map cells exactly to the original tree.
Nearest-centroid assignment would introduce an additional rule. The three null
models without effective clusters remain in the recurrence denominator of 30 splits.

## 20. Backup: SVM cell selection and recurrence

First select exactly the highest one percent of a full test donor's cell scores,
rounding up. At boundary ties, use the original FCS event number. Within this
selection, the distance from the learned donor threshold determines positive or
negative selection; equality belongs to neither direction. Negative selection
therefore does not mean separately selecting the lowest cell scores.

Positive frequency counts positive selections of the same cell, divided by every
outer test appearance of its donor. Negative frequency is analogous. Directions
are mutually exclusive within a split, but a cell may receive both directions
across different splits. Mean centered top-cell scores equal the donor score's
distance from its threshold. This decomposes a fixed selection; it is not a
causal effect of removing a cell.

Example: 101 cells give two top cells. At margins nine and eight and threshold
8.5, one cell is positive and one negative; the donor score equals the threshold.

## Sources and result version

The references for this revision are [Task 4 method explanations](../../results/AUFGABE_4_METHODENERKLAERUNGEN.md),
[Task 5 method explanations](../../results/AUFGABE_5_METHODENERKLAERUNGEN.md),
the existing notebooks, and the interpretation implementation.
Background: [CellCNN paper](https://doi.org/10.1038/ncomms14825),
[Citrus paper](https://doi.org/10.1073/pnas.1408792111), and
[Horowitz et al.](https://doi.org/10.1126/scitranslmed.3006702).
Result assets remain those saved on 9 September 2026. Their provenance is
documented in `data/provenance.json`.
