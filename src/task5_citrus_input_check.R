# Independently audit all native Citrus/FlowCore marker inputs, before arcsinh.
# Usage: Rscript src/task5_citrus_input_check.R SOURCE_ROOT OUTPUT_CSV
suppressPackageStartupMessages(library(citrus))
args <- commandArgs(TRUE)
stopifnot(length(args) == 2L)
root <- normalizePath(args[1])
data_root <- file.path(root, "NK_cell_dataset", "NK_cell_dataset")
markers <- scan(file.path(data_root, "NK_markers.csv"), what = character(), sep = ",", quiet = TRUE)
cells <- read.csv(file.path(root, "results", "tables", "task2_cells.csv"))
records <- list()
temporary <- tempfile(fileext = ".f64")
for (donor in sort(unique(cells$sample_id))) {
  path <- file.path(data_root, "NK_cell_dataset", "gated_alive", paste0(donor, "_alive.fcs"))
  fcs <- citrus.readFCS(path)
  channels <- match(markers, pData(flowCore::parameters(fcs))$desc)
  stopifnot(!anyNA(channels))
  values <- exprs(fcs)[, channels, drop = FALSE]
  writeBin(as.double(values), temporary, size = 8, endian = "little")
  records[[donor]] <- data.frame(donor_id = donor, n_cells = nrow(values), n_markers = ncol(values),
                                  source_fcs_md5 = unname(tools::md5sum(path)),
                                  native_marker_input_md5 = unname(tools::md5sum(temporary)))
}
unlink(temporary)
dir.create(dirname(args[2]), recursive = TRUE, showWarnings = FALSE)
write.csv(do.call(rbind, records), args[2], row.names = FALSE)
cat("Native Citrus input hashes exported for all 20 donors.\n")
