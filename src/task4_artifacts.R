# Konfiguration und numerisch wirksame Citrus-Cluster ohne Zusatzpakete prüfen.

check_run_config <- function(config_path, expected, artifact_paths, run_training) {
  present <- file.exists(artifact_paths)
  if (!file.exists(config_path) && !any(present)) {
    if (!run_training) stop("Keine Ergebnisse vorhanden; zuerst einen Lauf berechnen.")
    return(FALSE)
  }
  if (!file.exists(config_path)) {
    stop("Konfigurationsnachweis fehlt: ", basename(config_path),
         ". Altbestände können in 04e historisch ausgewertet werden. ",
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
