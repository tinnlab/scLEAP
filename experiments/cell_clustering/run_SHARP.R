
library(RhpcBLASctl)
RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::omp_set_num_threads(1)

Sys.setenv(OMP_NUM_THREADS = 1, OPENBLAS_NUM_THREADS = 1, MKL_NUM_THREADS = 1, VECLIB_MAXIMUM_THREADS = 1, NUMEXPR_NUM_THREADS = 1)

library(reticulate)
use_condaenv("/home/sy/miniconda3/envs/cell", required = TRUE)

library(SHARP)
library(SingleCellExperiment)
library(mclust)
library(ClusterR)
library(purrr)
library(anndata)
library(mclust)
library(parallel)
library(Matrix)

# Set a random seed for reproducibility
set.seed(1)

base_path = "./data/cellxgene"
path_to_save = "./results/clustering_results/SHARP/"

# create the directory if it does not exist
if (!dir.exists(path_to_save)) {
	dir.create(path_to_save, recursive = TRUE)
}
tissues_all <- list.files(base_path)
RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::omp_set_num_threads(1)

process_tissue <- function(tissue) {
  tryCatch({
    set.seed(1)

    # data_test_path <- file.path(base_path, tissue, "run_1", "test.h5ad")
    data_test_path <- file.path(base_path, tissue, "test.h5ad")

    data_test <- anndata::read_h5ad(data_test_path)
    count_matrix <- t(data_test$X)

    # Total-count normalization
    cell_sums <- Matrix::colSums(count_matrix)
    count_matrix_norm <- t(t(count_matrix) / cell_sums * 1e4)

    if (!inherits(count_matrix_norm, "dgCMatrix")) {
	count_matrix_log <- as(count_matrix_norm, "dgCMatrix")
	} else {
	count_matrix_log <- count_matrix_norm
	}
	## if max of count_matrix_log is greater than 20, log1p transformation
	if (max(count_matrix_log@x) > 20) {
		count_matrix_log@x <- log1p(count_matrix_log@x)
	}
    label <- data_test$obs["cell_type"]

    data <- count_matrix_log
    suppressMessages({
      suppressWarnings({
        res <- purrr::quietly(SHARP)(data, rN.seed = 1, n.cores=5, partition.ncells = 2000)$result
      })
    })

    cluster <- res$pred_clusters
    Clabel <- as.numeric(cluster)
    val <- sapply(c("adjusted_rand_index", "jaccard_index", "purity", "nmi"), function(x){
      round(external_validation(as.numeric(factor(label$cell_type)), Clabel, method = x), 2)
    })

    # Prepare a data.frame for saving to CSV
    cluster_df <- data.frame(
      cell = rownames(label),
      true_label = as.character(label$cell_type),
      cluster = Clabel
    )

    avg_silhouette_pred <- -1
    avg_silhouette_true <- -1

    results <- sprintf(
      "Tissue: %s\nMethod: SHARP\nARI: %.2f\nJaccard: %.2f\nPurity: %.2f\nNMI: %.2f\nnSilhouette cluster score: %.2f\nSilhouette: %.2f",
      tissue,
      val["adjusted_rand_index"],
      val["jaccard_index"],
      val["purity"],
      val["nmi"],
      avg_silhouette_pred,
      avg_silhouette_true
    )

    dir_path <- file.path(path_to_save, tissue)
    if (!dir.exists(dir_path)) {
      dir.create(dir_path, recursive = TRUE)
    }
    output_file <- file.path(dir_path, "results.txt")
    writeLines(results, output_file)

    # Write the cluster assignments and true labels to CSV
    cluster_csv_file <- file.path(dir_path, "clusters.csv")
    write.csv(cluster_df, cluster_csv_file, row.names = FALSE)

    return(paste("Results for", tissue, "have been written to:", output_file))

  }, error = function(e) {
    msg <- paste("Error processing tissue", tissue, ":", e$message)
    message(msg)
    return(msg)
  })
}

tissues <- list.dirs(base_path, full.names = FALSE, recursive = FALSE)

remain_tissues <- tissues[!file.exists(file.path(path_to_save, tissues, "results.txt"))]

print(paste("Remaining tissues to process:", length(remain_tissues)))

# num_cores <- 8
# results <- mclapply(remain_tissues, process_tissue, mc.cores = num_cores)

# cat(unlist(results), sep = "\n")

for (tissue in remain_tissues) {
  process_tissue(tissue)
}
# process_tissue(remain_tissues[2])
