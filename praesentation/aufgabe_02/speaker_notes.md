# Speaker notes · Task 2 · approximately nine minutes

Eight main slides are followed by ten reserve slides. The main talk does not require opening the reserve. Leave a short pause when introducing each figure. All numerical claims below refer to the provided 20-donor dataset, not the complete original Horowitz cohort.

## 1 · Data and question · 0:00–0:45

“We start with the mass-cytometry dataset from Horowitz and colleagues. Each observation is a cell described by 37 protein-marker intensities. Our provided dataset contains 20 labelled donors, and we use an equal sample of 2,000 NK cells per donor.

For this part of the project, we ask how PCA, t-SNE and UMAP display the same data, how their parameters change the maps, and how we can compare those maps. The later CellCNN analysis addresses prediction from cellular patterns. These visualizations alone do not show that CellCNN outperforms another method. Also, CMV status belongs to the donor: it does not tell us which individual cells are infected.”

## 2 · Preprocessing · 0:45–1:45

“The original intensity scales differ considerably between markers. We first apply arcsinh with cofactor five. Near zero the curve is almost linear; at larger intensities it compresses the range, and unlike a logarithm it accepts negative background measurements.

Next, each transformed marker is centred and divided by its standard deviation. This prevents one marker from dominating distances simply because of its numerical scale. All 37 markers pass the notebook’s variance filter.

The cofactor was chosen from a visual inspection, not a formal validation of an optimum. This pooled standardisation describes our exploratory sample; a classification pipeline must fit preprocessing using training donors only.”

## 3 · Sampling check · 1:45–2:45

“Equal donor contributions help avoid overrepresenting donors with more measured cells. But subsampling could still miss a phenotype of interest. Here we compare the fraction of NKG2C/CD57 double-positive cells in each donor’s full data with its fraction in the 2,000-cell sample.

This marker combination is associated with memory-like, sometimes called adaptive, NK cells. The word refers to altered immune-response properties after exposure, not proof that these individual cells are infected.

Each point is one donor. The points are close to the diagonal, which supports this particular sampling check. The positivity thresholds come from a simple Gaussian-mixture model. This is not a validated biological gate or a guarantee about every rare population.”

## 4 · PCA · 2:45–3:55

“PCA gives us a useful distinction between retaining information and displaying it. On the left, 29 components explain about 91.5 percent of the variance, so we retain these components for t-SNE and UMAP.

On the right, we display just the first two components, coloured by donor. Together they explain only 20.3 percent, even though the downstream input retains more than 90 percent. We should therefore not interpret the overlap in this picture as proof that there is no structure in the remaining dimensions.

The number of retained components is the main PCA choice investigated here. The displayed axes and choices such as scaling or whitening are separate decisions.”

## 5 · t-SNE · 3:55–5:15

“To keep the parameter study manageable, we use a fixed sample of 5,000 cells. Perplexity controls an effective neighbourhood scale. It is related to how broadly neighbour probabilities are distributed, rather than specifying an exact count of neighbours.

These examples show perplexities five, thirty, sixty and one hundred. The low-perplexity panel is more diffuse in this run, and the layouts change at larger values.

We select sixty because it has the highest saved Trustworthiness. That measure penalises false neighbours according to their distance ranks in the original representation. Its advantage over fifty is extremely small; we do not claim a robust optimum.

The second local metric, kNN preservation, counts exactly how many neighbours are shared. It was not the selection criterion here, and invalid saved sweep values have been omitted.”

## 6 · UMAP · 5:15–6:35

“UMAP separates two useful choices. The number of neighbours determines the scale of the graph built from the input. Minimum distance controls how tightly points can pack in the resulting map.

The four panels contrast small and large settings for both parameters. The smaller minimum distance permits compact islands, but these islands are not automatically distinct cell types. Increasing the number of neighbours changes the connectivity considered by the method.

The table lists the three highest saved Trustworthiness scores. We select five neighbours and minimum distance zero. This is the best setting under that particular criterion on this fixed sample, not a claim that it is universally best. The full sixteen-setting grid is in the reserve.”

## 7 · Quality comparison · 6:35–8:00

“These are the three maps of the same 40,000 cells. The table compares each map with the selected 29-PC representation, except for silhouette, which compares the CMV label groups within a map.

Trustworthiness and corrected kNN preservation both favour t-SNE for local structure. Pearson correlation of pairwise distances favours PCA. These results describe different properties, so there is no single overall winner implied by the table.

The final column asks whether cells carrying the two donor CMV labels are separated in each picture. All three silhouette values are close to zero. Although UMAP is numerically highest, that does not show strong clinical separation or predictive accuracy. Colours mark numerical ranks only.

Marker-expression overlays are available in the reserve if we want to explore biological hypotheses.”

## 8 · Pairwise comparison and conclusion · 8:00–9:00

“Finally, the assignment asks us to compare the maps directly with one another. We choose Procrustes disparity: after aligning position, rotation, reflection and scale, it measures how much global mismatch remains. Smaller is better.

PCA and t-SNE are the most similar pair by this measure. Corrected kNN overlap gives a different answer: t-SNE and UMAP share the most local neighbours. Global shape and local neighbourhood identity are different notions of similarity.

Our conclusion is that these maps help generate hypotheses, but their shapes and scores alone do not establish correct cell types or diagnostic performance. Those questions motivate the subsequent clustering and donor-level classification tasks.”

## Reserve guide

- **9:** t-SNE probabilities, perplexity, KL divergence and settings held fixed.
- **10:** Trustworthiness versus shared-neighbour fraction; the self-neighbour bug and its correction.
- **11:** UMAP weighted graph, conceptual objective and parameter roles.
- **12–13:** all sixteen UMAP settings, split into two readable views.
- **14–15:** original UMAP and t-SNE marker overlays.
- **16:** Shepard diagrams and the distinction between distance equality and perfect correlation after rescaling.
- **17:** Procrustes formula, row correspondence and invariances.
- **18:** references and verification scope.

## Answers to likely questions

**Did you rerun everything after fixing kNN?** No. Final kNN scores were recalculated from the existing coordinates. Procrustes and sampled distance correlations were checked. The models and parameter sweeps were not retrained. Invalid saved sweep kNN columns were removed. Trustworthiness and CMV silhouette are retained notebook outputs.

**Why not choose perplexity 50?** Perplexity 60 narrowly maximises the saved Trustworthiness used by the selection code. We did not establish that this very small difference is stable across seeds. The old kNN ranking was affected by the bug and is not a valid reason to change the selection.

**Why can Trustworthiness be high when neighbour overlap is low?** Trustworthiness weights false neighbours by their original ranks; overlap requires exact neighbour identities. Replacing a neighbour with a nearly adjacent rank may incur little Trustworthiness penalty but still reduces exact overlap.

**Is this an analysis of all peripheral blood cells?** No. This notebook starts from the provided gated NK subset. The broader gated-alive benchmark is a separate task. “Full data” in some retained plot titles means all 40,000 balanced analysis cells.

**Do colours mean the same thing everywhere?** In the selected sweep panels, CMV-negative is blue and CMV-positive is cyan. The common final-map figure uses blue and rust. The donor PCA plot has one colour per donor; marker overlays use continuous marker-specific scales. Read the caption or legend for each figure.
