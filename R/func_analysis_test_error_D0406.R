#' Analyze Test Error and Performance Metrics for a Model
#'
#' This function evaluates a trained model on a test dataset, calculating various performance metrics including autoencoder loss, outcome likelihood, classification accuracy, and AUC score for outcome prediction.
#'
#' @param args A list of arguments, typically containing configurations such as whether to use CUDA for computation.
#' @param model A trained model that will be evaluated. The model should have a \code{forward} method that supports an "outcome_logistic_regression" mode.
#' @param data_test A list representing the test dataset. This list will be updated with predicted cluster assignments.
#' @param dataloader_test An object representing the test data loader, which provides batches of test data to the model.
#'
#' @return A list containing the following performance metrics:
#' \itemize{
#'   \item \code{test_AE_loss}: The mean autoencoder loss (MSE) across all batches in the test dataset.
#'   \item \code{test_classifier_c_accuracy}: The classification accuracy of the model in predicting cluster assignments.
#'   \item \code{test_outcome_likelihood}: The mean binary cross-entropy loss for the outcome prediction.
#'   \item \code{outcome_auc_score}: The AUC (Area Under the Curve) score for the outcome prediction, which measures the model's ability to distinguish between positive and negative classes.
#' }
#'
#' @details
#' The \code{func_analysis_test_error_D0406} function performs the following steps:
#' \enumerate{
#'   \item Sets the model to evaluation mode to ensure proper handling of layers like dropout.
#'   \item Iterates over the test data batches provided by \code{dataloader_test}.
#'   \item For each batch, it performs a forward pass through the model using the "outcome_logistic_regression" mode, obtaining the encoded representations, decoded outputs, raw cluster predictions, and outcome predictions.
#'   \item Computes the mean squared error (MSE) for the autoencoder loss, binary cross-entropy (BCE) for the outcome prediction loss, and classification accuracy for cluster prediction.
#'   \item Accumulates the true outcomes and predicted probabilities for calculating the AUC score.
#'   \item Updates the \code{data_test} list with the predicted cluster assignments.
#'   \item Returns the average values of the computed metrics across all test batches.
#' }
#' @import torch
#' @export


func_analysis_test_error_D0406 <- function(args, model, data_test, dataloader_test) {
  model$eval()
  criterion_MSE <- nn_mse_loss()
  criterion_BCE <- nn_bce_loss()
  error_AE <- c()
  error_outcome_likelihood <- c()
  correct <- 0
  total <- 0
  outcome_true_y <- c()
  outcome_pred_prob <- c()
  cat("-----------------\n")

  #dataloader_test$reset()  # Reset the iterator
  # Create an iterator for the dataloader
  #dataloader_iterator <- 1 #dataloader_make_iter(dataloader_test)

  current_index <- 1
  batch_iter <- dataloader_test$.iter()  # Create an iterator

  while (current_index <= dataloader_test$.length()) {
    # To get the batch data, use an iterator
    batch  <- batch_iter$.next()  # Get the first batch

    # Check if the iterator is exhausted
    if (is.null(batch) || as.character(batch) == ".__exhausted__.") break

    batch_idx  <- batch_iter$.next()  # Get the first batch
    data_x <- batch_idx

    batch_idx <- batch$indices
    batch_data <- batch
    index <- batch$index
    batch_xvy <- batch[[1]]
    batch_c <- batch[[3]]

    data_x <- batch[[1]]
    data_v <- batch[[2]]
    target <- batch[[3]]

    data_x <- torch_tensor(data_x, requires_grad = FALSE)
    data_v <- torch_tensor(data_v, requires_grad = FALSE)
    target <- torch_tensor(target, requires_grad = FALSE)
    batch_c <- torch_tensor(batch_c, requires_grad = FALSE)

    if (args$cuda) {
      data_x <- data_x$cuda()
      data_v <- data_v$cuda()
      target <- target$cuda()
      batch_c <- batch_c$cuda()
    }

    # Reshape data_x to [1, batch_size, feature_size] for RNN input
    data_x <- data_x$unsqueeze(1)  # Add sequence dimension
    # Print the new shape to verify
    #print(data_x$size())  # Should print [1, 16, 4]

    # Forward pass
    output <- model$forward(x = data_x, function_name = "outcome_logistic_regression", demov = data_v)
    encoded_x <- output[[1]]
    decoded_x <- output[[2]]
    output_c_no_activate <- output[[3]]
    output_outcome <- output[[4]]

    # Trim decoded_x to match data_x
    decoded_x <- decoded_x[, 1:data_x$size(2), , drop = FALSE]

    # Print shapes to confirm they match
    #print(decoded_x$size())  # Should now match data_x$size()
    #print(data_x$size())

    loss_AE <- criterion_MSE(data_x, decoded_x)
    loss_outcome <- criterion_BCE(output_outcome, target$float())
    error_outcome_likelihood <- c(error_outcome_likelihood, as.numeric(loss_outcome$item()))
    error_AE <- c(error_AE, as.numeric(loss_AE$item()))

    # Classification accuracy
    predicted <- torch_max(output_c_no_activate$data(), 1)$indices
    #correct <- correct + sum(as.numeric(predicted == batch_c))
    total <- total + batch_c$size()

    outcome_true_y <- c(outcome_true_y, as.numeric(target$data()))
    outcome_pred_prob <- c(outcome_pred_prob, as.numeric(output_outcome$data()))

    current_index <- current_index + 1
  }

  test_classifier_c_accuracy <- correct / total
  test_AE_loss <- mean(error_AE)
  test_outcome_likelihood <- mean(error_outcome_likelihood)

  # Calculate AUC score
  outcome_auc_score <- auc(outcome_true_y, outcome_pred_prob)

  return(list(test_AE_loss = test_AE_loss,
              #test_classifier_c_accuracy = test_classifier_c_accuracy,
              test_outcome_likelihood = test_outcome_likelihood,
              outcome_auc_score = outcome_auc_score))
}
