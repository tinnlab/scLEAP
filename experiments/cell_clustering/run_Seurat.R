library(RhpcBLASctl)
RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::omp_set_num_threads(1)

Sys.setenv(OMP_NUM_THREADS = 1, OPENBLAS_NUM_THREADS = 1, MKL_NUM_THREADS = 1, VECLIB_MAXIMUM_THREADS = 1, NUMEXPR_NUM_THREADS = 1)

library(reticulate)
use_condaenv("/home/sy/miniconda3/envs/cell", required = TRUE)

library(Seurat)
library(anndata)
library(Matrix)
library(ggplot2)
library(dplyr)
library(cluster)
library(mclust)
library(parallel)
library(aricode)


base_path = "./data/cellxgene"
path_to_save = "./results/clustering_results/Seurat/"


# create path_to_save folder
if (!dir.exists(path_to_save)) {
  	dir.create(path_to_save, recursive = TRUE)
}

tissues_all <- list.files(base_path)
process_tissue <- function(tissue) {
  tryCatch({
    print(paste0("Processing tissue: ", tissue))
    set.seed(1)
    # data_test_path <- file.path(base_path, tissue, "run_1", "test.h5ad")
    data_test_path <- file.path(base_path, tissue, "test.h5ad")
    
    data_test <- anndata::read_h5ad(data_test_path)
    count_matrix <- t(data_test$X)
    
    if (is.null(rownames(count_matrix))) {
      rownames(count_matrix) <- paste0("Gene_", 1:nrow(count_matrix))
    }
    if (is.null(colnames(count_matrix))) {
      colnames(count_matrix) <- paste0("Cell_", 1:ncol(count_matrix))
    }
    
    seurat_obj <- CreateSeuratObject(counts = count_matrix, meta.data = data_test$obs)
    seurat_obj <- NormalizeData(seurat_obj)
    seurat_obj <- FindVariableFeatures(seurat_obj)
    seurat_obj <- ScaleData(seurat_obj)
    if (dim(seurat_obj)[2] > 100) {
      seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(object = seurat_obj), npcs = 50)
    } else {
      seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(object = seurat_obj), npcs = 20)
    }
    
    seurat_obj <- FindNeighbors(seurat_obj)
    print("Finding cluster")
    seurat_obj <- FindClusters(seurat_obj)
    print("")
    cluster_assignments <- Idents(seurat_obj)
    true_labels <- seurat_obj$cell_type
    
    # Save cluster assignments and true labels to CSV
    cluster_df <- data.frame(
      cell = names(cluster_assignments),
      cluster = as.character(cluster_assignments),
      label = as.character(true_labels)
    )
    
    dir_path <- file.path(path_to_save, tissue)
    if (!dir.exists(dir_path)) {
      dir.create(dir_path, recursive = TRUE)
    }
    cluster_csv_file <- file.path(dir_path, "clusters.csv")
    write.csv(cluster_df, cluster_csv_file, row.names = FALSE)
    
    pca_embeddings <- seurat_obj[["pca"]]@cell.embeddings[, 1:10]
    
    ari_score <- adjustedRandIndex(cluster_assignments, true_labels)
    num_clusters <- length(unique(cluster_assignments))
    nmi_score <- NMI(cluster_assignments, true_labels)
    ami_score <- AMI(cluster_assignments, true_labels)
    
    silhouette_score <- -1
    silhouette_cluster_score <- -1
    
    # try({
    #   silhouette <- silhouette(x = as.numeric(factor(true_labels)), dist = dist(pca_embeddings))
    #   silhouette_score <- mean(silhouette[, 3])
    #   silhouette_cluster <- silhouette(x = as.numeric(cluster_assignments), dist = dist(pca_embeddings))
    #   silhouette_cluster_score <- mean(silhouette_cluster[, 3])
    # }, silent = TRUE)
    
    results <- sprintf(
      "Tissue: %s\nMethod: louvain\nNumber of clusters: %d\nSilhouette score: %.17f\nARI: %.16f\nNMI: %.16f\nAMI: %.16f\nSilhouette cluster score: %.17f",
      tissue, num_clusters, silhouette_score, ari_score, nmi_score, ami_score, silhouette_cluster_score
    )
    
    output_file <- file.path(dir_path, "results.txt")
    writeLines(results, output_file)
    
    return(paste("Results for", tissue, "have been written to:", output_file, 
                 "\nCluster assignments saved to:", cluster_csv_file))
  }, error = function(e) {
    msg <- paste("Error processing tissue", tissue, ":", e$message)
    message(msg)
    return(msg)
  })
}

tissue_file <- "./data/valid_tissues.txt"
tissues <- readLines(tissue_file)

remain_tissues <- tissues[!file.exists(file.path(path_to_save, tissues, "results.txt"))]

# num_cores <- 8
# results <- mclapply(remain_tissues, process_tissue, mc.cores = num_cores)

## loop through tissues
for (tissue in remain_tissues) {
	process_tissue(tissue)
}


