# Speaker notes for Tasks 4 and 5

English speaking script for the nine main slides, planned for **8:00 minutes**.
The two backup slides are outside the regular talk. These timings are targets,
not a measured presentation duration. Read the quoted passages as the script;
the additional notes are delivery cues or material for questions.

| Slide | Duration | Finish at |
|---|---:|---:|
| 1 – Data and evaluation setup | 0:45 | 0:45 |
| 2 – CellCNN | 1:00 | 1:45 |
| 3 – Citrus | 0:45 | 2:30 |
| 4 – SVM cell score | 0:45 | 3:15 |
| 5 – SVM donor score | 1:00 | 4:15 |
| 6 – Performance comparison | 1:00 | 5:15 |
| 7 – Recurring subsets | 1:00 | 6:15 |
| 8 – Selected cells | 0:45 | 7:00 |
| 9 – Biological interpretation | 1:00 | 8:00 |

## 1. From cells to donor-level prediction — 45 seconds

> For Task 4, we predict a donor's CMV status. We use live, singlet PBMCs without
> an additional NK gate: 20 donors and 37 markers. We apply an ArcSinh
> transformation. CellCNN and SVM also use marker standardization, fitted only on
> training donors. Each split assigns 14 donors to training and six to testing.
> Model selection uses three inner folds within the training donors. All three
> methods share the same 30 outer splits. The key point is that donors are the
> independent observations: cells from the same donor always stay together.

Point to the split diagram. There is no need to repeat the general introduction
to CyTOF. The transformation shown on the slide uses a cofactor of five.

## 2. CellCNN learns filters for informative cells — 60 seconds

> CellCNN receives groups of 3,000 cells. Each filter learns a weighted combination
> of the 37 markers, and ReLU sets negative responses to zero. For each filter,
> we then average the highest one percent of responses. This allows a small,
> informative population to contribute to the prediction. The pooled values feed
> into the output layer to produce a CMV probability. Filters and predictions are
> learned jointly from donor labels. We generate 200 cell groups per training
> donor and select the best network using inner validation. Our PyTorch
> implementation follows the paper, with two adaptations: a fixed search over
> three, four, or five filters, and averaging five test-input probabilities instead
> of one. The selected network is not refitted on all 14 training donors.

Follow the diagram from left to right. Pooling produces the prediction; the
half-maximum threshold used later to define an interpretable subset is a separate
step. Each test input contains up to 20,000 cells; the fixed filter search
replaces the paper's random search.

## 3. Citrus predicts from population frequencies — 45 seconds

> Citrus works in two stages. First, it builds a hierarchical Ward tree using equal
> numbers of cells per training donor. Clusters containing at least 0.05 percent of
> the training cells provide a frequency for each donor. These frequencies enter
> an L1-regularized logistic regression. Many coefficients become zero; the
> remaining clusters are relevant to the model. New cells inherit memberships
> from their nearest training cell. We use the original R implementation with
> 10,000 cells per donor. This cell count and the 30 repetitions are computational
> compromises relative to the paper. Population discovery precedes supervised
> classification.

The tree contains nested clusters. Its features are therefore not a table of
mutually exclusive cell types. We use the original Citrus R implementation v0.8.

## 4. A linear SVM assigns a score to each cell — 45 seconds

> Our third method is a linear single-cell SVM. Each training cell receives its
> donor's CMV label. This is weak supervision: a positive donor does not consist
> entirely of disease-associated cells. The diagram illustrates the idea with two
> markers; our model uses all 37. The SVM learns a linear decision function whose
> value is a cell's margin. Higher values indicate stronger evidence in the model's
> positive direction. This score is not a probability. A single-cell SVM already
> appears as a baseline in the paper. Our adaptation concerns the next step:
> combining cell margins into a donor score.

The point colours represent donor labels, not validated states of individual
cells.

## 5. Our adaptation: aggregate the highest cell margins — 60 seconds

> To predict a donor's status, we sort all cell margins from that test donor and
> average the highest one percent. For a thousand cells, we keep ten. In this toy
> example, those ten margins range from 2.1 to 3.0, giving a mean of 2.55. At an
> illustrative threshold of 1.8, the prediction is positive. In the actual
> benchmark, we first choose the SVM parameter C using mean inner donor AUC.
> We learn the threshold from inner out-of-fold donor scores by maximizing
> sensitivity plus specificity. We then refit the SVM on all 14 training donors.
> The outer test set determines neither C nor the threshold. This top-one-percent
> aggregation is our project adaptation, fixed before testing.

The diagram's widths are not proportional to cell counts. AUC uses the continuous
donor score, not the binary prediction. The example threshold of 1.8 is not a fitted
benchmark result.

## 6. Performance across 30 shared donor splits — 60 seconds

> Here we compare all three methods on exactly the same 30 splits. Each dot is the
> ROC-AUC on six test donors, and each box shows the interquartile range. CellCNN
> and SVM both reach a median AUC of 0.875; Citrus reaches 0.625. Variation across
> splits is substantial, particularly for Citrus. We also report average precision
> and balanced accuracy. CellCNN and SVM share the same medians for these measures
> here, although their individual predictions can differ. Balanced accuracy also
> depends on the decision threshold. These results do not establish general
> superiority: there are still only 20 independent donors, and the repeated splits
> overlap. The quartiles describe variation across splits, not confidence intervals.

BA uses a threshold of 0.5 for CellCNN/Citrus and an inner-trained threshold for
the SVM.

If asked why AUC values occur in coarse steps: two positive and four negative test
donors yield only eight positive–negative pairs. Tied scores contribute half a
correctly ranked pair.

## 7. Which CellCNN and Citrus subsets recur? — 60 seconds

> Task 5 asks which cell subsets are associated with the models. For CellCNN, we
> divide each filter response by its maximum on the original training reference.
> Normalized responses above 0.5 define the subset; its mean marker profile is the
> centroid. For Citrus, we use saved centroids of clusters with nonzero regression
> coefficients. We group similar centroids separately for each method and count
> how many distinct splits contain each group. Groups occurring in at least six
> of 30 splits appear on the t-SNE map. The most frequent CellCNN group occurs in
> all 30 splits, and the most frequent Citrus group in 18. Both are positively
> associated with CMV. These points are projected training centroids; matching
> colours across panels do not identify the same population.

This map contains **10,000 `gated_alive` cells**, 500 per donor, at perplexity 30.
It differs from the 40,000-NK-cell map in the existing Task 2 slides on `main`.
The display space was established exploratively using all donors; it does not
enter classifier evaluation.

Details for questions: average linkage, cosine distance, and a distance cutoff of
0.4. Transferring this rule from the reference code's filter grouping to subset
centroids is a documented adaptation. The three Citrus null models remain in the
denominator. Multiple members of a group in one split count only once.

## 8. Which cells does the SVM repeatedly select? — 45 seconds

> On the left, one representative CellCNN filter selects 87 map cells. This is an
> exploratory application to all donors. On the right, 93 map cells were selected
> positively by the SVM at least once. A selected cell must be among the highest
> one percent of margins in its full test donor and exceed the saved threshold.
> Colour shows how often this happens across that donor's test appearances,
> ranging from four to 16 appearances per donor. This cell frequency and the
> group recurrence on the previous slide use different denominators. They are not
> directly comparable effect sizes.

The map is a sample: 93 is the number of highlighted map cells, not the size of
the full selected population. Selection uses all cells of each test donor, not
the top one percent of the displayed map alone.

## 9. What do the marker profiles support? — 60 seconds

> The heatmap now includes the SVM. CellCNN and Citrus still show saved group
> representatives. For the SVM, we average positively selected cells within each
> test visit, then visits within donors, and finally all 20 donors equally. There
> are 168 nonempty visits out of 180. CellCNN shows elevated NKG2C and CD57 with
> NK-associated markers, compatible with the paper's memory-like NK profile.
> CD3 and CD57 suggest a T-associated pattern for the Citrus representative.
> The SVM also shows elevated NKG2C, CD57, and NK-associated markers, with CD3
> slightly above the reference mean. The shared z-scale makes marker values
> comparable, but the subset definitions differ. These profiles support
> biological hypotheses, not validated cell-type assignments.

The SVM profile uses all cells from the respective outer test donors in splits
0–29, regardless of their true CMV labels. It is not restricted to the 93 map
cells. Twelve test visits have no positive selection: their profiles are missing,
not zero. All 20 donors have at least one nonempty visit. Nonempty visits receive
equal weight within each donor, and donors receive equal weight in the final
profile. This describes marker expression conditional on positive selection,
not how frequently selection occurs.

The stored representatives remain CellCNN split 9/filter 2 and Citrus split
25/cluster 139891. The SVM row is not a group representative. Do not interpret
z-values as positive or negative gates. Mean profiles can summarize heterogeneous
cells; CD3 is not zero in the CellCNN representative either.

## 10. Backup: hyperparameters and paper deviations

Open only if asked. CellCNN retains the selected inner-fold network, trained on
nine or ten donors. Citrus and SVM perform their final fits on 14 donors. Citrus
fits its inner trees within each fold, but its original lambda grid uses all 14
outer training donors. Outer test donors remain excluded from fitting.

Citrus standardizes frequency features within its glmnet fits; it does not apply
additional z-scoring to the ArcSinh-transformed markers themselves.

CellCNN validation ties are resolved by AUC, loss, fewer filters, and fold order.
SVM ties favour the smaller C. Citrus `cv.min` minimizes inner classification
error, favouring stronger regularization in a tie. The 37 provided markers differ
from the 36 reported in the paper. Different cell samples and GPU/CPU hardware
limit a direct runtime comparison.

For interpretation, the average-linkage/cosine/cutoff-0.4 rule was transferred
from filter grouping in the reference code. It is not a confirmed reproduction
of the paper's NK-centroid grouping parameters.

## 11. Backup: CellCNN and SVM across 100 splits

This comparison includes only CellCNN and SVM. Both again reach a median AUC of
0.875. Median balanced accuracy is 0.750 for CellCNN and 0.625 for SVM. The 600
test predictions per method are repeated predictions for the same 20 donors.
Keep these results separate from the three-method comparison over 30 splits;
they do not form a single shared table or ranking with Citrus.

## Sources and result version

Background: [CellCNN paper](https://doi.org/10.1038/ncomms14825),
[Citrus paper](https://doi.org/10.1073/pnas.1408792111), and
[Horowitz et al.](https://doi.org/10.1126/scitranslmed.3006702).
The slides use the saved Task 4/5 results from 9 September 2026. Input checksums
and exported assets are documented in `data/provenance.json`.
