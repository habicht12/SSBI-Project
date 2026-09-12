source("src/task4_artifacts.R")

expect_error <- function(expression) {
  failed <- tryCatch({force(expression); FALSE}, error = function(e) TRUE)
  stopifnot(failed)
}
directory <- tempfile("citrus_config_test_")
dir.create(directory)
config_path <- file.path(directory, "config.rds")
artifacts <- file.path(directory, c("predictions.csv", "selection.csv", "clusters.csv"))
config <- list(cofactor = 5, tolerance = 1e-10, inputs = c(splits = "original"))
stopifnot(!check_run_config(config_path, config, artifacts, TRUE))
expect_error(check_run_config(config_path, config, artifacts, FALSE))
invisible(file.create(artifacts))
expect_error(check_run_config(config_path, config, artifacts, TRUE))
saveRDS(config, config_path)
stopifnot(check_run_config(config_path, config, artifacts, TRUE))
for (training in c(TRUE, FALSE)) {
  changed <- config; changed$cofactor <- 10
  expect_error(check_run_config(config_path, changed, artifacts, training))
  changed <- config; changed$inputs <- c(splits = "changed inner folds")
  expect_error(check_run_config(config_path, changed, artifacts, training))
}
unlink(artifacts[3])
expect_error(check_run_config(config_path, config, artifacts, TRUE))

coefficients <- matrix(c(1, 0, 1e-15, 1e-10, -0.2), ncol = 1,
  dimnames = list(c("(Intercept)", "cluster 1 abundance", "cluster 2 abundance",
                    "cluster 3 abundance", "cluster 4 abundance"), NULL))
stopifnot(identical(effective_cluster_ids(coefficients, 1e-10), 4L))
selection <- data.frame(split_id = 0:2, selected_cluster_count = c(1, 1, 0))
predictions <- selection[rep(1:3, each = 6), ]
clusters <- data.frame(split_id = c(0, 1), cluster_id = c(1, 2),
                       marker = "CD3", coefficient = c(1e-15, -0.2))
clean <- normalize_cluster_tables(predictions, selection, clusters, 1e-10)
stopifnot(identical(clean$selection$selected_cluster_count, c(0L, 1L, 0L)),
          identical(clean$predictions$selected_cluster_count, rep(c(0L, 1L, 0L), each = 6)),
          nrow(clean$clusters) == 1L)
empty <- normalize_cluster_tables(predictions, selection, clusters[FALSE, ], 1e-10)
stopifnot(all(empty$predictions$selected_cluster_count == 0L))
unlink(directory, recursive = TRUE)
cat("Citrus-Konfiguration und Nulltoleranz: alle Tests bestanden.\n")
