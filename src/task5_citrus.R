# Aufgabe 5: finalen Citrus-Clusterraum rekonstruieren, niemals Regression/CV neu fitten.

cluster_cell_scores <- function(memberships, coefficients, event_count) {
  # Hierarchische Cluster überlappen; auch gegenläufige Gewichte müssen sich addieren.
  stopifnot(length(memberships) == length(coefficients), all(is.finite(coefficients)))
  scores <- numeric(event_count)
  for (i in seq_along(coefficients)) {
    members <- memberships[[i]]
    stopifnot(!anyDuplicated(members), all(members >= 1L), all(members <= event_count))
    scores[members] <- scores[members] + coefficients[[i]]
  }
  scores
}

centered_logit_errors <- function(probabilities, mean_scores) {
  stopifnot(length(probabilities) == length(mean_scores),
            all(is.finite(probabilities)), all(probabilities > 0 & probabilities < 1),
            all(is.finite(mean_scores)))
  logits <- qlogis(probabilities)
  abs((logits - mean(logits)) - (mean_scores - mean(mean_scores)))
}

run_task5_citrus <- function(root, output_directory, split_ids) {
  suppressPackageStartupMessages(library(citrus))
  tables <- file.path(root, "results", "tables")
  data_root <- file.path(root, "NK_cell_dataset", "NK_cell_dataset")
  directory <- file.path(data_root, "NK_cell_dataset", "gated_alive")
  markers_path <- file.path(data_root, "NK_markers.csv")
  labels_path <- file.path(data_root, "NK_fcs_samples_with_labels.csv")
  splits_path <- file.path(tables, "task4_donor_splits.csv")
  markers <- as.character(read.csv(text = paste(readLines(markers_path, warn = FALSE), collapse = "\n"),
                                   header = FALSE)[1, ])
  splits <- read.csv(splits_path)
  cells <- read.csv(file.path(tables, "task2_cells.csv"))
  profiles <- read.csv(file.path(tables, "task4_citrus_clusters_gated_alive_full.csv"))
  selection <- read.csv(file.path(tables, "task4_citrus_selection_gated_alive_full.csv"))
  predictions <- read.csv(file.path(tables, "task4_citrus_predictions_gated_alive_full.csv"))
  config <- readRDS(file.path(tables, "task4_citrus_predictions_gated_alive_full.config.rds"))
  input_paths <- c(splits_path, labels_path, markers_path,
                   sort(list.files(directory, pattern = "\\.fcs$", full.names = TRUE)))
  input_hashes <- setNames(as.character(tools::md5sum(input_paths)), basename(input_paths))
  stopifnot(identical(config$inputs, input_hashes), length(markers) == 37L,
            config$parameters$gate == "gated_alive", config$parameters$transform_cofactor == 5,
            config$parameters$file_sample_size == 1000,
            config$parameters$minimum_cluster_size_fraction == 0.05,
            config$parameters$coefficient_tolerance == 1e-10,
            config$parameters$citrus_commit == packageDescription("citrus")$RemoteSha,
            config$parameters$rclusterpp_commit == packageDescription("Rclusterpp")$RemoteSha,
            identical(config$parameters$r_version, R.version.string))
  for (package in names(config$parameters$package_versions)) {
    stopifnot(as.character(packageVersion(package)) == config$parameters$package_versions[[package]])
  }
  stopifnot(!anyDuplicated(predictions[c("split_id", "donor_id")]),
            !anyDuplicated(profiles[c("split_id", "cluster_id", "marker")]),
            !anyDuplicated(selection$split_id), setequal(selection$split_id, 0:99),
            nrow(predictions) == 600L, all(abs(profiles$coefficient) > 1e-10),
            all(profiles$gate == "gated_alive"), all(profiles$transform_cofactor == 5),
            all(selection$selected_cluster_count >= 0L))

  # Jede Datei nur einmal lesen; spätere Stichproben reproduzieren citrus.readFCSSet.
  donors <- sort(unique(splits$donor_id))
  donor_data <- setNames(vector("list", length(donors)), donors)
  for (donor in donors) {
    fcs <- citrus.readFCS(file.path(directory, paste0(donor, "_alive.fcs")))
    descriptions <- as.character(Biobase::pData(flowCore::parameters(fcs))$desc)
    invalid <- which(nchar(descriptions) < 3 | is.na(descriptions))
    descriptions[invalid] <- flowCore::colnames(fcs)[invalid]
    stopifnot(!anyDuplicated(descriptions), all(markers %in% descriptions))
    values <- flowCore::exprs(fcs)[, match(markers, descriptions), drop = FALSE]
    colnames(values) <- markers
    stopifnot(all(is.finite(values)))
    donor_data[[donor]] <- asinh(values / 5)
  }
  sample_donors <- function(ids, seed) {
    set.seed(seed)
    parts <- lapply(seq_along(ids), function(file_id) {
      values <- donor_data[[ids[[file_id]]]]
      # R-Sampling und 1-basierte Ereignisnummern wie citrus.readFCSSet.
      indices <- sort(sample(seq_len(nrow(values)), 1000L))
      cbind(values[indices, , drop = FALSE], fileEventNumber = indices, fileId = file_id)
    })
    result <- list(data = do.call(rbind, parts), fileNames = paste0(ids, "_alive.fcs"),
                   fileIds = matrix(seq_along(ids), ncol = 1L, dimnames = list(NULL, "unstim")))
    class(result) <- "citrus.combinedFCSSet"
    result
  }
  all_cells <- list()
  all_checks <- list()
  all_cluster_checks <- list()
  for (split_id in split_ids) {
    split <- splits[splits$split_id == split_id, ]
    train_ids <- sort(split$donor_id[split$outer_partition == "train"])
    test_ids <- sort(split$donor_id[split$outer_partition == "test"])
    selected <- selection[selection$split_id == split_id, ]
    saved <- predictions[predictions$split_id == split_id, ]
    saved <- saved[match(test_ids, saved$donor_id), ]
    stopifnot(length(train_ids) == 14L, length(test_ids) == 6L,
              length(intersect(train_ids, test_ids)) == 0L, nrow(selected) == 1L,
              selected$r_seed == unique(split$split_seed) %% .Machine$integer.max,
              all(saved$split_seed == unique(split$split_seed)),
              all(saved$y_true == split$label[match(test_ids, split$donor_id)]),
              all(saved$file_sample_size == 1000), all(saved$decision_threshold == 0.5))
    selected_profiles <- profiles[profiles$split_id == split_id, ]
    cluster_ids <- sort(unique(selected_profiles$cluster_id))
    stopifnot(length(cluster_ids) == selected$selected_cluster_count)
    plot_cells <- cells[cells$sample_id %in% test_ids, ]
    plot_values <- do.call(rbind, lapply(test_ids, function(donor) {
      local_cells <- plot_cells[plot_cells$sample_id == donor, ]
      stopifnot(nrow(local_cells) == 500,
                all(local_cells$cell_id == paste("gated_alive", donor, local_cells$event_index, sep = ":")))
      donor_data[[donor]][local_cells$event_index + 1L, , drop = FALSE]
    }))
    stopifnot(identical(unique(plot_cells$sample_id), test_ids))
    if (length(cluster_ids) == 0L) {
      raw_scores <- numeric(nrow(plot_cells))
      means <- numeric(6L)
      arithmetic_error <- 0
      maximum_centroid_error <- 0
      eligible_count <- NA_integer_
    } else {
      train <- sample_donors(train_ids, as.integer(selected$r_seed))
      clustering <- citrus.cluster(train, markers)
      eligible <- citrus.selectClusters(clustering, minimumClusterSizePercent = 0.05)
      stopifnot(all(cluster_ids %in% eligible), length(eligible) == selected$eligible_cluster_count)
      eligible_count <- length(eligible)
      coefficients <- numeric(length(cluster_ids))
      centroid_errors <- numeric(length(cluster_ids))
      memberships <- clustering$clusterMembership[cluster_ids]
      for (index in seq_along(cluster_ids)) {
        group <- selected_profiles[selected_profiles$cluster_id == cluster_ids[[index]], ]
        group <- group[match(markers, group$marker), ]
        stopifnot(nrow(group) == 37, !anyNA(group$marker), length(unique(group$coefficient)) == 1,
                  length(memberships[[index]]) >= 0.05 * nrow(train$data))
        coefficients[[index]] <- group$coefficient[[1]]
        centroid <- colMeans(train$data[memberships[[index]], markers, drop = FALSE])
        centroid_errors[[index]] <- max(abs(centroid - group$centroid))
      }
      maximum_centroid_error <- max(centroid_errors)
      stopifnot(maximum_centroid_error < 1e-10)
      training_scores <- cluster_cell_scores(memberships, coefficients, nrow(train$data))
      test <- sample_donors(test_ids, as.integer(selected$r_seed + 1L))
      # Originale native Nächste-Trainingszelle-Zuordnung, keine Zentroidapproximation.
      combined <- rbind(plot_values, test$data[, markers, drop = FALSE])
      neighbors <- citrus:::citrus.assignToCluster(combined, train$data[, markers, drop = FALSE],
                                                  rep(1L, nrow(train$data)))
      raw_scores <- training_scores[neighbors[seq_len(nrow(plot_cells))]]
      test_neighbors <- tail(neighbors, nrow(test$data))
      test_scores <- training_scores[test_neighbors]
      means <- as.numeric(tapply(test_scores, test$data[, "fileId"], mean))
      # Unabhängige Feature-Rechnung aus den nativen Mitgliedschaftslisten.
      assignments <- lapply(clustering$clusterMembership, citrus:::citrus.mapNeighborsToCluster,
                            nearestNeighborMap = test_neighbors)
      features <- citrus.calculateFeatures(test, assignments, cluster_ids, featureType = "abundances")
      arithmetic_error <- max(abs(as.numeric(features %*% coefficients) - means))
      stopifnot(arithmetic_error < 1e-12)
    }
    errors <- centered_logit_errors(saved$score, means)
    # Alle 15 paarweisen Differenzen zusätzlich prüfen; kein Intercept schätzen.
    pair_error <- max(abs(outer(qlogis(saved$score), qlogis(saved$score), "-") - outer(means, means, "-")))
    stopifnot(max(errors) < 1e-8, pair_error < 1e-8)
    all_cells[[length(all_cells) + 1L]] <- data.frame(
      split_id = split_id, cell_id = plot_cells$cell_id, raw_score = raw_scores)
    all_checks[[length(all_checks) + 1L]] <- data.frame(
      method = "Citrus", split_id = split_id, donor_id = test_ids,
      saved_score = saved$score, reconstructed_score = NA_real_, score_error = NA_real_,
      arithmetic_error = errors, pairwise_logit_error = pair_error,
      feature_identity_error = arithmetic_error,
      # Bei fehlendem Intercept nur die gespeicherte Schwellenentscheidung kontrollieren.
      decision_matches = saved$y_pred == as.integer(saved$score >= 0.5))
    all_cluster_checks[[length(all_cluster_checks) + 1L]] <- data.frame(
      split_id = split_id, selected_cluster_count = length(cluster_ids),
      eligible_cluster_count = eligible_count, maximum_centroid_error = maximum_centroid_error,
      tree_reconstructed = length(cluster_ids) > 0L, null_model = length(cluster_ids) == 0L)
    cat("Citrus: Split", split_id, "rekonstruiert und geprüft.\n")
    flush.console()
  }
  write.csv(do.call(rbind, all_cells), file.path(output_directory, "cells.csv"), row.names = FALSE)
  write.csv(do.call(rbind, all_checks), file.path(output_directory, "checks.csv"), row.names = FALSE)
  write.csv(do.call(rbind, all_cluster_checks), file.path(output_directory, "clusters.csv"), row.names = FALSE)
}

if (sys.nframe() == 0L) {
  arguments <- commandArgs(trailingOnly = TRUE)
  if (length(arguments) != 3L) stop("Aufruf: task5_citrus.R PROJECT_ROOT OUTPUT_DIRECTORY SPLIT_IDS")
  split_ids <- as.integer(strsplit(arguments[[3]], ",", fixed = TRUE)[[1]])
  stopifnot(length(split_ids) > 0L, !anyNA(split_ids), !anyDuplicated(split_ids), all(split_ids %in% 0:99))
  run_task5_citrus(arguments[[1]], arguments[[2]], split_ids)
}
