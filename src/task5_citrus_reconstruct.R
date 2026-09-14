# Reconstruct the original final Citrus training tree; do not refit classifiers.
# Usage: Rscript src/task5_citrus_reconstruct.R SOURCE_ROOT CACHE_DIR [0,1,...]
# Large intermediate files stay in the ignored results/tables cache.

suppressPackageStartupMessages(library(citrus))
options(mc.cores = 1)

reconstruct_citrus <- function(root, cache, split_ids) {
  root <- normalizePath(root)
  dir.create(cache, recursive = TRUE, showWarnings = FALSE)
  cache <- normalizePath(cache)
  source(file.path(root, "src", "task4_artifacts.R"))
  validate_comparison_config(root)
  stopifnot(packageDescription("citrus")$RemoteSha == "d02baae544abdc403704aaceb75d1e7931a0331c",
            packageDescription("Rclusterpp")$RemoteSha == "a07380683ce7a6849af8ec27db6439ea3a707890")
  tables <- file.path(root, "results", "tables")
  data_root <- file.path(root, "NK_cell_dataset", "NK_cell_dataset")
  markers <- scan(file.path(data_root, "NK_markers.csv"), what = character(), sep = ",", quiet = TRUE)
  splits <- read.csv(file.path(tables, "task4_donor_splits.csv"))
  selection <- read.csv(file.path(tables, "task4_citrus_selection_gated_alive_full.csv"))
  profiles <- read.csv(file.path(tables, "task4_citrus_clusters_gated_alive_full.csv"), check.names = FALSE)
  map <- read.csv(file.path(tables, "task2_cells.csv"))
  stopifnot(all(split_ids %in% 0:29), length(markers) == 37L, !anyDuplicated(split_ids))
  input_files <- file.path(tables, c("task4_donor_splits.csv", "task4_citrus_selection_gated_alive_full.csv",
                                    "task4_citrus_clusters_gated_alive_full.csv",
                                    "task4_citrus_predictions_gated_alive_full.config.rds", "task2_cells.csv"))
  fingerprint <- as.list(setNames(unname(tools::md5sum(input_files)), basename(input_files)))
  fingerprint$exporter_md5 <- unname(tools::md5sum(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))))

  for (sid in split_ids) {
    directory <- file.path(cache, sprintf("split_%02d", sid))
    manifest_path <- file.path(directory, "manifest.json")
    if (file.exists(manifest_path)) {
      old <- jsonlite::read_json(manifest_path)
      if (!identical(old$fingerprint, fingerprint)) stop("Stale Citrus cache: ", directory)
      actual <- as.list(setNames(unname(tools::md5sum(file.path(directory, names(old$output_md5)))), names(old$output_md5)))
      if (!identical(actual, old$output_md5)) stop("Citrus cache checksum mismatch: ", directory)
      cat("Reuse verified Citrus reconstruction, split", sid, "\n")
      next
    }
    started <- proc.time()[[3]]
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    split <- splits[splits$split_id == sid, ]
    train_ids <- sort(split$donor_id[split$outer_partition == "train"])
    test_ids <- sort(split$donor_id[split$outer_partition == "test"])
    selected <- profiles[profiles$split_id == sid, ]
    model <- selection[selection$split_id == sid, ]
    stopifnot(length(train_ids) == 14, length(test_ids) == 6, !any(train_ids %in% test_ids), nrow(model) == 1)
    coefficient <- unique(selected[c("cluster_id", "coefficient")])
    positive_ids <- coefficient$cluster_id[coefficient$coefficient > 1e-10]
    output_files <- character()
    max_centroid_error <- 0
    if (length(positive_ids)) {
      set.seed(model$r_seed)
      training <- citrus.readFCSSet(
        dataDirectory = file.path(data_root, "NK_cell_dataset", "gated_alive"),
        fileList = data.frame(unstim = paste0(train_ids, "_alive.fcs")), fileSampleSize = 10000L,
        transformColumns = markers, transformCofactor = 5, useChannelDescriptions = TRUE)
      stopifnot(nrow(training$data) == 140000L)
      cat("Reconstruct final training tree for split", sid, "\n"); flush.console()
      tree <- citrus:::citrus.cluster.hierarchical(training$data[, markers])
      positive <- rep(FALSE, nrow(training$data))
      membership_rows <- list()
      for (cluster_id in coefficient$cluster_id) {
        members <- citrus:::citrus.traverseMergeOrder(cluster_id, tree$merge)
        saved <- selected[selected$cluster_id == cluster_id, ]
        saved <- saved[match(markers, saved$marker), ]
        stopifnot(nrow(saved) == length(markers), all(saved$marker == markers))
        error <- max(abs(colMeans(training$data[members, markers, drop = FALSE]) - saved$centroid))
        if (error > 1e-11) stop("Reconstructed training cluster differs from saved centroid: split ", sid,
                               ", cluster ", cluster_id, ", max error ", error)
        max_centroid_error <- max(max_centroid_error, error)
        if (cluster_id %in% positive_ids) positive[members] <- TRUE
        membership_rows[[as.character(cluster_id)]] <- data.frame(cluster_id = cluster_id, n_members = length(members),
                                                                  coefficient = saved$coefficient[1], max_centroid_error = error)
      }
      writeBin(as.double(training$data[, markers]), file.path(directory, "training.f64"), size = 8, endian = "little")
      writeBin(as.raw(positive), file.path(directory, "positive.u8"))
      write.csv(data.frame(donor_id = train_ids[as.integer(training$data[, "fileId"])],
                           event_index = as.integer(training$data[, "fileEventNumber"]) - 1L),
                file.path(directory, "training_events.csv"), row.names = FALSE)
      write.csv(do.call(rbind, membership_rows), file.path(directory, "cluster_checks.csv"), row.names = FALSE)
      saveRDS(tree, file.path(directory, "tree.rds"))
      # Native Citrus nearest-neighbour mapping of deterministic, actual test-map
      # cells. This independently audits the accelerated full-donor mapping.
      native_rows <- list()
      for (donor in test_ids) {
        fcs <- citrus.readFCS(file.path(data_root, "NK_cell_dataset", "gated_alive", paste0(donor, "_alive.fcs")))
        values <- exprs(fcs)
        channels <- match(markers, pData(flowCore::parameters(fcs))$desc)
        chosen <- head(map[map$sample_id == donor, ], 64L)
        query <- asinh(values[chosen$event_index + 1L, channels, drop = FALSE] / 5)
        nearest <- citrus:::citrus.assignToCluster(query, training$data[, markers], rep(1L, nrow(training$data)))
        native_rows[[donor]] <- data.frame(cell_id = chosen$cell_id, donor_id = donor,
                                           event_index = chosen$event_index, nearest_train_index = nearest - 1L,
                                           positive = positive[nearest])
      }
      write.csv(do.call(rbind, native_rows), file.path(directory, "native_map_checks.csv"), row.names = FALSE)
      output_files <- c("training.f64", "positive.u8", "training_events.csv", "cluster_checks.csv", "tree.rds", "native_map_checks.csv")
      rm(training, tree); gc()
    }
    manifest <- list(split_id = sid, train_donors = train_ids, test_donors = test_ids,
                     r_seed = model$r_seed, n_training_cells = 140000L, markers = markers,
                     positive_cluster_ids = as.list(positive_ids),
                     max_centroid_error = max_centroid_error, null_positive_selection = !length(positive_ids),
                     elapsed_seconds = proc.time()[[3]] - started, fingerprint = fingerprint,
                     package_versions = list(R = R.version.string, citrus = as.character(packageVersion("citrus")),
                                             Rclusterpp = as.character(packageVersion("Rclusterpp"))),
                     output_md5 = as.list(setNames(unname(tools::md5sum(file.path(directory, output_files))), output_files)))
    jsonlite::write_json(manifest, manifest_path, pretty = TRUE, auto_unbox = TRUE, digits = 17)
    cat("Citrus reconstruction complete, split", sid, "in", manifest$elapsed_seconds, "seconds\n"); flush.console()
  }
}

if (sys.nframe() == 0L) {
  args <- commandArgs(TRUE)
  stopifnot(length(args) %in% 2:3)
  split_ids <- if (length(args) == 3) as.integer(strsplit(args[3], ",", fixed = TRUE)[[1]]) else 0:29
  reconstruct_citrus(args[1], args[2], split_ids)
}
