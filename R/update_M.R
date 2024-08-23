#' Update Cluster Embeddings in Matrix M
#'
#' This function updates the matrix \code{M}, which contains cluster embeddings, based on the current cluster assignments and representations in the training data.
#'
#' @param data_train A list containing the training data. The list must include the following components:
#' \itemize{
#'   \item \code{M}: A matrix where each column corresponds to a cluster's embedding vector.
#'   \item \code{n_cat}: An integer representing the number of clusters (or categories).
#'   \item \code{rep}: A matrix of representations (embeddings) for the training data.
#'   \item \code{C}: An integer vector indicating the cluster assignment for each training sample.
#' }
#'
#' @return The function updates the \code{M} matrix in-place within the \code{data_train} list, reflecting the new average embeddings for each cluster.
#'
#' @details
#' The \code{update_M} function works as follows:
#' \enumerate{
#'   \item It first checks that the number of columns in the \code{M} matrix matches the number of clusters (\code{n_cat}). If not, an error is thrown.
#'   \item It then initializes a list of lists, \code{dict_c_embedding}, where each sublist will hold the embeddings for samples belonging to a specific cluster.
#'   \item The function iterates over all samples, adding their representations to the corresponding cluster sublist in \code{dict_c_embedding}.
#'   \item For each cluster, the function computes the mean of the embeddings across all samples assigned to that cluster.
#'   \item The computed mean embedding is then used to update the corresponding column in the \code{M} matrix.
#' }
#' @import torch
update_M <- function(data_train) {
  # Check if the number of columns in M matches n_cat
  if (ncol(data_train$M) != data_train$n_cat) {
    stop("Invalid M!")
  }

  dict_c_embedding <- vector("list", data_train$n_cat)
  representations <- data_train$rep
  list_C <- as.integer(data_train$C)

  for (i in seq_len(data_train$n_cat)) {
    dict_c_embedding[[i]] <- list()
  }

  for (i in seq_along(list_C)) {
    cp <- list_C[i]
    dict_c_embedding[[cp + 1]] <- c(dict_c_embedding[[cp + 1]], list(representations[i, , drop = FALSE]))
  }

  for (i in seq_len(data_train$n_cat)) {
    c_key <- i
    if (length(dict_c_embedding[[c_key]]) == 0) {
      next
    }
    embed_list <- torch_stack(dict_c_embedding[[c_key]], dim = 1)
    embed_mean_dim0 <- torch_mean(embed_list, dim = 2)
    data_train$M[, c_key] <- embed_mean_dim0
  }
  cat("    update M!\n")
}
