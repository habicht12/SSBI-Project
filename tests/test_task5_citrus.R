# Kleine unabhängige Handrechnungen; kein Modellfit und keine FCS-Dateien.
source("src/task5_citrus.R")
scores <- cluster_cell_scores(list(c(1L, 2L), c(2L, 3L)), c(2, -2), 4L)
stopifnot(identical(scores, c(2, 0, -2, 0)))
stopifnot(identical(cluster_cell_scores(list(), numeric(), 3L), numeric(3L)))
means <- c(-2, 0, 1, 3)
probabilities <- plogis(1.5 + means)
stopifnot(max(centered_logit_errors(probabilities, means)) < 1e-12)
stopifnot(max(centered_logit_errors(probabilities, rev(means))) > 1)
stopifnot(inherits(try(centered_logit_errors(c(0, 1), c(1, 2)), silent = TRUE), "try-error"))
suppressPackageStartupMessages(library(citrus))
training <- list(data = matrix(c(0, 0.1, 10, 10.1), ncol = 1,
                               dimnames = list(NULL, "marker")))
clustering <- citrus.cluster(training, "marker")
cluster_ids <- which(lengths(clustering$clusterMembership) == 2L)
# Native Cluster-IDs sind Listenpositionen, keine Namen: numerischer Zugriff ist nötig.
memberships <- clustering$clusterMembership[cluster_ids]
stopifnot(length(cluster_ids) == 2L, all(lengths(memberships) == 2L))
train_scores <- cluster_cell_scores(memberships, c(2, -2), 4L)
neighbors <- citrus:::citrus.assignToCluster(matrix(c(0.09, 9.9), ncol = 1), training$data, rep(1, 4))
stopifnot(identical(as.integer(neighbors), c(2L, 3L)))
stopifnot(setequal(train_scores[neighbors], c(-2, 2)))
cat("Citrus-Aufgabe-5-Rechentests bestanden.\n")
