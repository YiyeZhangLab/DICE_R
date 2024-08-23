require(torch)
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
