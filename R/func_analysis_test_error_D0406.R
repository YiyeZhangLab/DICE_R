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

  dataloader_test$reset()  # Reset the iterator

  while (dataloader_test$has_next()) {
    #### <-------------
    #batch <- dataloader_test$next()
    batch_idx <- batch[[1]]
    batch_data <- batch[[2]]
    index <- batch_data[[1]]
    batch_xvy <- batch_data[[2]]
    batch_c <- batch_data[[3]]

    data_x <- batch_xvy[[1]]
    data_v <- batch_xvy[[2]]
    target <- batch_xvy[[3]]

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

    # Forward pass
    output <- model$forward(x = data_x, function_name = "outcome_logistic_regression", demov = data_v)
    encoded_x <- output[[1]]
    decoded_x <- output[[2]]
    output_c_no_activate <- output[[3]]
    output_outcome <- output[[4]]

    loss_AE <- criterion_MSE(data_x, decoded_x)
    loss_outcome <- criterion_BCE(output_outcome, target$float())
    error_outcome_likelihood <- c(error_outcome_likelihood, as.numeric(loss_outcome$item()))
    error_AE <- c(error_AE, as.numeric(loss_AE$item()))

    # Classification accuracy
    predicted <- torch_max(output_c_no_activate$data(), 1)$indices
    correct <- correct + sum(as.numeric(predicted == batch_c))
    total <- total + batch_c$size(0)

    outcome_true_y <- c(outcome_true_y, as.numeric(target$data()))
    outcome_pred_prob <- c(outcome_pred_prob, as.numeric(output_outcome$data()))

    data_test$pred_C[index] <- predicted$cpu()
  }

  test_classifier_c_accuracy <- correct / total
  test_AE_loss <- mean(error_AE)
  test_outcome_likelihood <- mean(error_outcome_likelihood)

  # Calculate AUC score
  outcome_auc_score <- auc(outcome_true_y, outcome_pred_prob)

  return(list(test_AE_loss = test_AE_loss, test_classifier_c_accuracy = test_classifier_c_accuracy, test_outcome_likelihood = test_outcome_likelihood, outcome_auc_score = outcome_auc_score))
}
