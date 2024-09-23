#' Update Test Set Representations, Cluster Assignments, and Cluster Centers
#'
#' This function updates the representations, cluster assignments, and cluster centers for a test dataset based on a trained model and a reference training dataset.
#'
#' @param args A list of arguments, typically containing configurations such as the number of hidden features (\code{n_hidden_fea}) and whether to use CUDA (\code{cuda}).
#' @param model A trained model, typically an autoencoder, that will be used to generate new representations for the test data.
#' @param data_test A list representing the test dataset. This list will be updated with new representations, cluster assignments, and cluster centers.
#' @param dataloader_test An object representing the test data loader, which provides batches of test data to the model.
#' @param data_train A list representing the training dataset, which contains reference cluster centers and other attributes needed for updating the test dataset.
#'
#' @return The function updates the following attributes within the \code{data_test} list:
#' \itemize{
#'   \item \code{rep}: The new representations (embeddings) for the test dataset.
#'   \item \code{n_cat}: The number of clusters, set to match the training dataset.
#'   \item \code{M}: The cluster centers, copied from the training dataset.
#'   \item \code{pred_C}: The predicted cluster assignments for each test sample, based on the updated representations and cluster centers.
#' }
#'
#' @details
#' The \code{update_testset_R_C_M_K} function performs the following steps:
#' \enumerate{
#'   \item Initializes a tensor, \code{final_embed}, to store the embeddings for the test dataset.
#'   \item Sets the model to evaluation mode.
#'   \item Iterates through the test data batches using \code{dataloader_test}, processing each batch through the model to obtain the embeddings.
#'   \item Updates the \code{rep} attribute of \code{data_test} with the new embeddings.
#'   \item Updates \code{data_test$n_cat} and \code{data_test$M} to match the training dataset.
#'   \item For each test sample, computes the closest cluster center (based on Euclidean distance) and assigns the corresponding cluster to \code{pred_C}.
#' }
#' @import torch

update_testset_R_C_M_K <- function(args, model, data_test, dataloader_test, data_train) {
  cat("-----------------\n")
  cat("    update_testset_R_C_M_K\n")

  # Initialize final_embed tensor
  final_embed <- torch_randn(nrow(data_test), args$n_hidden_fea, dtype = torch_float())

  # Set the model to evaluation mode
  model$eval()

  dataloader_test$reset()  # Reset the iterator

  while (dataloader_test$has_next()) {
    #### <-------------
    #batch <- dataloader_test$next()
    #batch_idx <- batch[[1]]
    #batch_data <- batch[[2]]
    #index <- batch_data[[1]]
    #batch_xvy <- batch_data[[2]]
    #batch_c <- batch_data[[3]]

    batch_iter <- dataloader_test$.iter()  # Create an iterator
    batch_idx  <- batch_iter$.next()  # Get the first batch
    data_x <- batch_idx


    #data_x <- batch_xvy[[1]]
    #data_v <- batch_xvy[[2]]
    #target <- batch_xvy[[3]]

    #data_x <- torch_tensor(data_x, requires_grad = FALSE)
    #data_v <- torch_tensor(data_v, requires_grad = FALSE)
    #target <- torch_tensor(target, requires_grad = FALSE)

    #if (args$cuda) {
    #  data_x <- data_x$cuda()
    #  data_v <- data_v$cuda()
    #  target <- target$cuda()
    #}

    # Perform a forward pass through the model in autoencoder mode
    output <- model$forward(data_x, "autoencoder")
    enc <- output[[1]]
    pred <- output[[2]]

    # Get the embeddings
    embed <- enc$data$cpu()[,1,,drop=FALSE]
    final_embed[index] <- embed
  }

  # Update data_test attributes
  data_test$rep <- final_embed
  cat("        update data_test R!\n")

  data_test$n_cat <- data_train$n_cat
  cat("        update data_test n_cat\n")

  data_test$M <- data_train$M
  cat("        update data_test M\n")

  # Update data_test.C
  representations <- data_test$rep
  pred_C <- torch_zeros(nrow(data_test), dtype = torch_int())

  for (i in seq_len(nrow(representations))) {
    embed <- representations[i,]
    trans_embed <- embed$view(c(embed$size(), 1))
    xj <- torch_norm(trans_embed - data_train$M, dim = 0)
    new_cluster <- torch_argmin(xj)
    pred_C[i] <- new_cluster
  }

  data_test$pred_C <- pred_C
  cat("        update pred data_test C\n")
}
