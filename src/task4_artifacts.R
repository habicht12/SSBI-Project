# Konfiguration und numerisch wirksame Citrus-Cluster ohne Zusatzpakete prüfen.

check_run_config <- function(config_path, expected, artifact_paths, run_training) {
  present <- file.exists(artifact_paths)
  if (!file.exists(config_path) && !any(present)) {
    if (!run_training) stop("Keine Ergebnisse vorhanden; zuerst einen Lauf berechnen.")
    return(FALSE)
  }
  if (!file.exists(config_path)) {
    stop("Konfigurationsnachweis fehlt: ", basename(config_path),
         ". Auch der Comparison benötigt einen passenden Nachweis. ",
         "Für einen neuen Lauf die alten Artefakte separat sichern und ",
         "aus den aktiven Ergebnispfaden verschieben; sie werden nicht überschrieben.")
  }
  if (!identical(readRDS(config_path), expected)) {
    stop("Gespeicherte Ergebnisse passen nicht zur aktuellen Konfiguration. ",
         "Alte Artefakte separat sichern und einen getrennten Lauf beginnen.")
  }
  if (!all(present)) {
    stop("Unvollständiger Zwischenstand: ",
         paste(basename(artifact_paths[!present]), collapse = ", "))
  }
  TRUE
}

effective_cluster_ids <- function(coefficients, tolerance) {
  stopifnot(is.finite(tolerance), tolerance >= 0, all(is.finite(coefficients)))
  feature_names <- rownames(coefficients)
  selected <- grepl("^cluster [0-9]+ abundance$", feature_names) &
    abs(coefficients[, 1]) > tolerance
  as.integer(sub("^cluster ([0-9]+) abundance$", "\\1", feature_names[selected]))
}

normalize_cluster_tables <- function(predictions, selection, clusters, tolerance) {
  # Auch alte, bereits gespeicherte Profile können ohne Neutraining bereinigt werden.
  stopifnot(all(is.finite(clusters$coefficient)))
  clusters <- clusters[abs(clusters$coefficient) > tolerance, , drop = FALSE]
  counts <- table(unique(clusters[c("split_id", "cluster_id")])$split_id)
  for (name in c("predictions", "selection")) {
    value <- get(name)
    count <- as.integer(counts[as.character(value$split_id)])
    count[is.na(count)] <- 0L
    value$selected_cluster_count <- count
    assign(name, value)
  }
  list(predictions = predictions, selection = selection, clusters = clusters)
}

validate_comparison_config <- function(root) {
  # Nur Nachweise lesen; keine FCS-Daten einlesen und keine Modelle fitten.
  tables <- file.path(root, "results", "tables")
  data <- file.path(root, "NK_cell_dataset", "NK_cell_dataset")
  path <- file.path(tables, "task4_citrus_predictions_gated_alive_full.config.rds")
  expected <- readRDS(path)
  expected$schema_version <- 1L
  parameters <- list(
    method = "citrus", gate = "gated_alive", run_mode = "full",
    transform_cofactor = 5, file_sample_size = 10000L,
    minimum_cluster_size_fraction = 0.0005, coefficient_tolerance = 1e-10,
    inner_folds = 3L, clustering = "hierarchical", feature_type = "abundances",
    model = "glmnet", alpha = 1, standardize_features = TRUE,
    selection = "cv.min", score_rounding_digits = 15L, decision_threshold = 0.5,
    citrus_commit = "d02baae544abdc403704aaceb75d1e7931a0331c",
    rclusterpp_commit = "a07380683ce7a6849af8ec27db6439ea3a707890"
  )
  expected$parameters[names(parameters)] <- parameters
  inputs <- c(file.path(tables, "task4_donor_splits.csv"),
              file.path(data, "NK_fcs_samples_with_labels.csv"),
              file.path(data, "NK_markers.csv"),
              sort(list.files(file.path(data, "NK_cell_dataset", "gated_alive"),
                              pattern = "\\.fcs$", full.names = TRUE)))
  expected$inputs <- setNames(as.character(tools::md5sum(inputs)), basename(inputs))
  notebook <- jsonlite::read_json(file.path(root, "notebooks", "04d_citrus.ipynb"))
  definitions <- new.env(parent = environment())
  eval(parse(text = paste0(unlist(notebook$cells[[9]]$source), collapse = "")), definitions)
  functions <- c(mget(c("read_citrus_files", "build_fixed_fold_clustering",
                       "extract_selected_cluster_profiles", "run_citrus_split"), definitions),
                 list(effective_cluster_ids, normalize_cluster_tables))
  expected$implementation <- unname(lapply(functions, function(f) paste(deparse(body(f)), collapse = "\n")))
  artifacts <- file.path(tables, paste0("task4_citrus_", c("predictions", "selection", "clusters"),
                                     "_gated_alive_full.csv"))
  check_run_config(path, expected, artifacts, run_training = FALSE)
  cat("Citrus-Konfiguration, Eingabeprüfsummen und Trainingscode stimmen.\n")
}
