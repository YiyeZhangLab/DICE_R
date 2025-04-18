#' Evaluate Autoencoder Model on Test Data
#'
#' This function evaluates the performance of an autoencoder model on a test dataset, calculating the mean squared error (MSE) between the original inputs and the reconstructed outputs.
#'
#' @param args A list of arguments, typically containing configurations such as whether to use CUDA for computation.
#' @param model A trained autoencoder model that will be evaluated. The model is expected to have a `forward` method that takes input data and a mode (e.g., "autoencoder").
#' @param dataloader_test A list representing the test data loader. Each element in the list should be a batch containing test data that the model will evaluate.
#'
#' @return A numeric value representing the mean test error (MSE) across all batches in the test dataset.
#'
#' @details
#' The \code{test_AE} function performs the following steps:
#' \enumerate{
#'   \item Initializes the mean squared error (MSE) loss criterion.
#'   \item Sets the model to evaluation mode to ensure layers like dropout are appropriately handled.
#'   \item Iterates over the test data batches provided by \code{dataloader_test}.
#'   \item For each batch, the function:
#'   \itemize{
#'     \item Extracts the data, ensuring it is in numeric format and reshapes it as necessary.
#'     \item Converts the data into torch tensors, moving them to CUDA if specified.
#'     \item Performs a forward pass through the model in autoencoder mode, obtaining both the encoded representation and the reconstructed output.
#'     \item Computes the MSE loss between the input and the reconstructed output.
#'     \item Accumulates the loss for each batch.
#'   }
#'   \item Computes the mean of the accumulated losses to return the overall test error.
#' }
#' @import torch
#' @export


test_AE <- function(args, model, dataloader_test) {
  # Define the MSE loss criterion
  criterion_MSE <- nn_mse_loss()

  test_error <- c()
  cat("-----------------\n")

  # Set the model to evaluation mode
  model$eval()

  suppressWarnings( {
    for (batch in 1:length(dataloader_test)) {
      # To get the batch data, use an iterator
      batch_iter <- dataloader_test$.iter()  # Create an iterator
      batch_idx  <- batch_iter$.next()  # Get the first batch
      data_x <- batch_idx[[1]]
      data_x <- torch_tensor(data_x, requires_grad = FALSE)
      data_x <- data_x$unsqueeze(1)

      # Perform a forward pass through the model in autoencoder mode
      output <- model$forward(data_x, "autoencoder")
      enc <- output[[1]]
      pred <- output[[2]]

      # Compute the loss
      loss <- criterion_MSE(data_x, pred)

      # Append the loss to the test_error list
      test_error <- c(test_error, as.numeric(loss$item()))
    }
  } )

  # Compute the mean test error
  test_AE_error <- mean(test_error)
  return(test_AE_error)
}
