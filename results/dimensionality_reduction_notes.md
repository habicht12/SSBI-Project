Generate latex slides out of this notebook, which answer the task: 
Dimensionality reduction: Download the provided dataset. Visualize the dataset with PCA/tSNE/UMAP. What are the parameters that you can choose for each dimensionality reduction technique? What is the impact of these parameters on the visualization. Document example visualizations. What quantitative measures could you use to compare the difference of the visualizations? Choose one measure and assess it for all pairs of visualizations.

Context: Horowitz et al, 2013 paper, notebook: 02_dimensionality_reduction_clean, task, maybe slides from task 3

Sldes:
1. a slide which explains where the data cames from (Horowitz et al., 2013, CyTOF, 37 Markers), what we wanted to show (How CellCnn outperforms other methods to find CMV infectioned cells)
2. A slide about the arcsinh transformation we did with an plot of these function and also why and that we z-scored (preprocessing)
3. Slide for testing the 2,000 cells/donor subsets if they are representative for NKG2C+/CD57+ "adaptive" NK cells the paper associates with CMV infection. Explan the adaptive term. Take the plot from cell 17. 
4. PCA: test for number of components, with 29 more than 90% explained variance, include elbowplot (cell 20), PCA colored by donor, but first two PCs explain only about 20% together.
5. hyperparameter tuning: therfore a 5,000 cells subsample was taken (runtime and RAM)
t-SNE: Perplexity (how many neighbors t-SNE
tries to preserve per point (conceptually similar to UMAP's `n_neighbors`)) Show effekt of perplexity with plots (maybe just take 5, 30, 60, 100) we have choosen 60
based on trustworthiness and knn-preservation, one sentence about them each
backup slide about how tsne works with more detailed explanation of the other parameters, and why we have not taken them into consideration.
backup slide about trustworthiness and knn-preservation with equations
UMAP: short explanation n_neighbors	min_dist	
first few rows from outputtable and plot from cell 33, quite large, Selected t-SNE perplexity: 60
Selected UMAP params: n_neighbors=5, min_dist=0.0
backupslide for UMAP parameters and short how it works
6. comparison: plot, output plots from cell 41
short explanations non-neighbor-based metrics
table from cell 43, with colorcoded, for each metric best(green), worst (red), or so 
backup slide with shepard embedding and explanation and perfect plot for comparison
7. **Pairwise comparison between the three embeddings themselves** (as opposed to each vs. the
high-D reference) -- for these we use the symmetric measures (kNN-preservation, Procrustes
disparity), table, explanation of results

Also I would like to have a short report about what was done with the most important figures and maybe a few others in the appendix which should work in the whole context of all tasks. And I want a longer report with ausführlichen explanations of metrics formulas and how things work (TSNE, UMAP), parameters...