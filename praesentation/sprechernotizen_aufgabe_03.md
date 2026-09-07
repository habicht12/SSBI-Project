**Speaker notes for Task 3 · approximately 8–9 minutes**

The eight main slides form the talk. Backup slides 9–11 are intended for questions. The suggested timings add up to approximately 8 minutes 50 seconds.

**Slide 1 · Question and data · 45 seconds**

“For Task 3, we compare K-means, hierarchical clustering and Leiden. Our question is which partition reflects biologically meaningful cell populations. We analyse 20,000 live PBMCs using 37 markers. Each of the 20 donors contributes exactly 1,000 cells. The key point is that a good clustering score alone is not sufficient to identify biological cell types.”

Explain PBMC as peripheral blood mononuclear cell if needed. The analysis uses the alive files, rather than a sample restricted to previously gated NK cells.

**Slide 2 · Workflow and selection · 75 seconds**

The arcsinh transformation reduces the influence of large intensity differences; each marker is then standardised. Every method receives the same matrix. Scores are calculated on that matrix, not on UMAP coordinates.

K-means and the four hierarchical variants test seven cluster counts. Leiden tests nine combinations of neighbour count and resolution. This gives 44 configurations. Leiden resolution does not directly specify the number of clusters.

Each configuration is refitted ten times on a subset retaining 80% of cells within each donor. Scaling is also refitted. A mean cluster stability of at least 0.6 determines eligibility; among eligible configurations, we select the lowest Davies–Bouldin index separately for each method. Lower DB values indicate more favourable geometric separation.

**Slide 3 · Results table · 75 seconds**

The table shows the winner within each method, not an automatically selected overall winner. K-means produces four clusters with mean stability 0.995. Leiden produces 18 clusters using 30 neighbours and resolution 1.2, with stability 0.683. Their DB values of 2.363 and 2.361 are very close; we do not treat this as an established quality difference.

The last column is particularly revealing: single, average and complete linkage each put more than 99.9% of cells in a single cluster. Ward produces six more substantial groups, but is considerably less stable than K-means.

**Slide 4 · The score's limitation · 50 seconds**

“Single linkage has the lowest DB score and stability 1. Yet its solution consists of 19,999 cells plus one individual cell. This does not provide a useful partition for identifying cell types.”

An isolated event can remain isolated when other cells are removed. High stability therefore does not resolve this problem. The second cluster would be invisible in a bar drawn to scale, so its count is shown separately. Do not imply that small clusters are always artefacts.

**Slide 5 · Shared UMAP · 60 seconds**

Both images show the same cells at the same UMAP coordinates. The left image colours four broad groups; the right image divides the structure into 18 groups. Colours are assigned separately within each method; matching colours across the two images do not indicate matching cell memberships.

UMAP is used only for display. It provides no independent proof of separation in the original feature space. The finer Leiden solution generates candidate subgroups that need to be assessed individually.

**Slide 6 · Individual clusters versus the mean · 75 seconds**

Each point is a cluster's median Jaccard overlap over ten repeats. All four K-means clusters lie close to 1. Three of six Ward clusters and four of 18 Leiden clusters lie below 0.6. Leiden clusters C15 and C16 are particularly unstable, at approximately 0.11 and 0.12.

The dashed line marks 0.6 for comparison. In the notebook, this threshold is applied only to the mean of the per-cluster medians. It is not an individual-cluster filter. A configuration can therefore remain eligible even when some of its groups are unstable.

**Slide 7 · Biological interpretation · 90 seconds**

These are cautious hypotheses based on relative marker profiles. K-means C0 has relatively higher CD19 and HLA-DR and lower CD3. Leiden C3 shows a similar profile, consistent with a B-cell-like group. K-means C3 and Leiden C4 show NK-associated markers and are candidates for NK-cell-like groups.

Do not claim that both methods have therefore grouped the same cells: this comparison concerns their profiles. Nor should all four K-means clusters be described as four identified cell types. The remaining profiles are insufficiently specific.

Arrows indicate the difference between the cluster median and the sample median, scaled by the sample IQR. They do not represent experimentally defined marker positivity. Reliable annotations would require additional markers and assessment of coexpression in individual cells. Horowitz et al. provides biological context; these assignments remain our own preliminary interpretations.

**Slide 8 · Reasoned assessment · 60 seconds**

“For a robust broad partition, K-means is our starting point in this run. Leiden provides finer hypotheses, but several subgroups are unstable. The strongly degenerate single, average and complete solutions contribute little to the cell-type question, while Ward is less stable.”

A final biological decision requires validated marker annotations and checks for donor and acquisition-date effects. Fresh initial cell samples are particularly important for small groups. We can therefore answer the quantitative question and justify our current assessment without claiming a biological winner that the available evidence has not established.

**Backup slides**

- **9:** The comparison with the 200-cell run changes both sample size and the k grid. The reference uses the correct stability threshold of 0.6. Differences cannot be attributed to cell count alone.
- **10:** Explain Jaccard overlap using exactly the same retained cell IDs. Cluster labels may change; the calculation finds the best overlap in membership. The reference is not ground truth. Scaling is refitted in each repeat.
- **11:** Sources and reproducibility. The executed notebook contains all tables and full figures. Compiling the presentation does not require rerunning the 20-minute clustering analysis.
