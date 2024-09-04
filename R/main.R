#' Main Function for Training and Evaluating a Deep Learning Model
#'
#' This function orchestrates the training and evaluation of a deep learning model, specifically for autoencoder-based representation learning followed by clustering and classification. It handles data loading, model training, testing, and saving the best model based on performance criteria.
#'
#' @param args A list of arguments that configure the model training and evaluation process. The list should include:
#' \itemize{
#'   \item \code{seed}: An integer value for setting the random seed for reproducibility.
#'   \item \code{input_path}: A character string specifying the path to the input data directory.
#'   \item \code{filename_train}: A character string specifying the filename of the training dataset.
#'   \item \code{filename_test}: A character string specifying the filename of the test dataset.
#'   \item \code{n_input_fea}: An integer specifying the number of input features.
#'   \item \code{n_hidden_fea}: An integer specifying the number of hidden features in the model.
#'   \item \code{lstm_layer}: An integer specifying the number of LSTM layers in the model.
#'   \item \code{lstm_dropout}: A numeric value specifying the dropout rate for the LSTM layers.
#'   \item \code{K_clusters}: An integer specifying the number of clusters for the k-means clustering.
#'   \item \code{n_dummy_demov_fea}: An integer specifying the number of dummy demographic features.
#'   \item \code{cuda}: A logical value indicating whether to use CUDA (GPU acceleration) for model training.
#'   \item \code{lr}: A numeric value specifying the learning rate for the optimizer.
#'   \item \code{init_AE_epoch}: An integer specifying the number of epochs for training the autoencoder.
#'   \item \code{iter}: An integer specifying the number of iterations for the main optimization process.
#'   \item \code{epoch_in_iter}: An integer specifying the number of epochs in each iteration of the main optimization process.
#'   \item \code{lambda_AE}: A numeric value specifying the weight of the autoencoder loss in the overall loss function.
#'   \item \code{lambda_classifier}: A numeric value specifying the weight of the classification loss in the overall loss function.
#'   \item \code{lambda_outcome}: A numeric value specifying the weight of the outcome prediction loss in the overall loss function.
#'   \item \code{lambda_p_value}: A numeric value specifying the weight of the p-value loss in the overall loss function.
#' }
#'
#' @details
#' The \code{main} function executes the following steps:
#' \enumerate{
#'   \item Sets the random seed for reproducibility.
#'   \item Loads and preprocesses the training and test datasets.
#'   \item Initializes the model, optimizer, and loss functions.
#'   \item Trains an autoencoder for representation learning, saving the model and plotting loss curves.
#'   \item Performs k-means clustering on the learned representations and reassigns cluster labels based on outcome ratios.
#'   \item Iteratively trains the model for clustering, classification, and outcome prediction, optimizing the combined loss function.
#'   \item Saves the best model based on the outcome prediction likelihood and checks for p-value significance.
#'   \item Outputs and saves relevant training metrics, including loss curves and model checkpoints.
#' }
#' @import torch ggplot2

main <- function(args) {
  set.seed(args$seed)

  # Load data
  data_train <- yf_dataset_withdemo(args$input_path, args$filename_train, args$n_hidden_fea)
  dataloader_train <- dataloader(tensor_dataset(list(data_train$data_x, data_train$data_v, data_train$data_y)), batch_size = 1, shuffle = TRUE, drop_last = TRUE)
  data_test <- yf_dataset_withdemo(args$input_path, args$filename_test, args$n_hidden_fea)
  dataloader_test <- dataloader(tensor_dataset(list(data_test$data_x, data_test$data_v, data_test$data_y)), batch_size = 1, shuffle = FALSE, drop_last = TRUE)

  # Algorithm 2 model
  model <- model_2(args$n_input_fea, args$n_hidden_fea, args$lstm_layer, args$lstm_dropout, args$K_clusters, args$n_dummy_demov_fea, args$cuda)
  optimizer <- optim_adam(model$parameters(), lr = args$lr)
  criterion_MSE <- nn_mse_loss()
  criterion_BCE <- nn_bce_loss()
  criterion_CrossEntropy <- nn_cross_entropy_loss()

  print(model)
  if (args$cuda) {
    model <- model$cuda()
  }

  # Autoencoder, initialize the representation
  print("/////////////////////////////////////////////////////////////////////////////")
  print("part 1: train AE and for representation initialization")

  args$output_path <- paste0("./hn_", args$n_hidden_fea, "_K_", args$K_clusters)
  if (dir.exists(args$output_path)) {
    unlink(args$output_path, recursive = TRUE)
  }
  dir.create(args$output_path)

  part1_foldername <- paste0(args$output_path, "/part1_AE_nhidden_", args$n_hidden_fea)
  if (dir.exists(part1_foldername)) {
    unlink(part1_foldername, recursive = TRUE)
  }
  dir.create(part1_foldername)

  loss_list <- c()
  train_AE_loss_list <- c()
  test_AE_loss_list <- c()
  number_reassign_list <- c()
  random_state_list <- c()

  for (epoch in seq_len(args$init_AE_epoch)) {
    error <- c()
    print("-----------------")
    model$train()
    dataloader_train$reset()
    while (dataloader_train$has_next()) {
      #### <-------------
      #batch <- dataloader_train$next()
      index <- batch[[1]]
      batch_xvy <- batch[[2]]
      batch_c <- batch[[3]]

      data_x <- batch_xvy[[1]]
      data_v <- batch_xvy[[2]]
      target <- batch_xvy[[3]]

      data_x <- torch_tensor(data_x, requires_grad = FALSE)
      data_v <- torch_tensor(data_v, requires_grad = FALSE)
      target <- torch_tensor(target, requires_grad = FALSE)

      if (args$cuda) {
        data_x <- data_x$cuda()
        data_v <- data_v$cuda()
        target <- target$cuda()
      }

      output <- model$forward(x = data_x, function_name = "autoencoder")
      enc <- output[[1]]
      pred <- output[[2]]

      optimizer$zero_grad()
      loss <- criterion_MSE(data_x, pred)
      loss$backward()
      optimizer$step()
      error <- c(error, as.numeric(loss$item()))
    }
    loss_list <- c(loss_list, mean(error))

    train_AE_loss <- mean(error)
    test_AE_loss <- test_AE(args, model, dataloader_test)
    print(sprintf("Epoch: %s, train AE loss: %s, test AE loss: %s.", epoch, train_AE_loss, test_AE_loss))

    train_AE_loss_list <- c(train_AE_loss_list, train_AE_loss)
    test_AE_loss_list <- c(test_AE_loss_list, test_AE_loss)

    torch_save(model$state_dict(), file.path(part1_foldername, paste0('AE_model_', epoch, '.pt')))
    print("    Saving AE models")
  }

  # Plotting loss curves
  df_loss <- data.frame(epoch = seq_len(args$init_AE_epoch), train_AE_loss = train_AE_loss_list, test_AE_loss = test_AE_loss_list)
  p <- ggplot(df_loss, aes(x = epoch)) +
    geom_line(aes(y = train_AE_loss, color = "train_AE_loss")) +
    geom_line(aes(y = test_AE_loss, color = "test_AE_loss_list")) +
    labs(title = "Autoencoder Loss", x = "Epoch", y = "Loss") +
    scale_color_manual("", values = c("train_AE_loss" = "green", "test_AE_loss_list" = "blue"))
  ggsave(file.path(part1_foldername, "part1_loss_AE.png"), plot = p)

  print("part 1, initial done!")
  print("////////////////////////////////////////////////////////////////////////////////////")
  print("part 2, start optimization the main loss")
  part2_foldername <- paste0(args$output_path, "/part2_AE_nhidden_", args$n_hidden_fea)
  if (dir.exists(part2_foldername)) {
    unlink(part2_foldername, recursive = TRUE)
  }
  dir.create(part2_foldername)

  iter_train_negloglikeli_list <- c()
  iter_test_negloglikeli_list <- c()
  iter_train_classifier_acc_list <- c()
  iter_test_classifier_acc_list <- c()
  iter_train_AE_list <- c()
  iter_test_AE_list <- c()
  iter_train_auc_list <- c()
  iter_test_auc_list <- c()

  min_test_negloglikeli_record <- 1000000
  saved_iter <- -1
  saved_iter_list <- c()

  for (iter_i in seq_len(args$iter)) {
    print("****************************************************************************************")
    print(paste("iter_i=", iter_i))

    # Part 2, clustering
    final_embed <- torch_randn(nrow(data_train$data_x), args$n_hidden_fea, dtype = torch_float())
    model$eval()
    dataloader_train$reset()
    while (dataloader_train$has_next()) {
      #### <-------------
      #batch <- dataloader_train$next()
      index <- batch[[1]]
      batch_xvy <- batch[[2]]
      batch_c <- batch[[3]]

      data_x <- batch_xvy[[1]]
      data_v <- batch_xvy[[2]]
      target <- batch_xvy[[3]]

      data_x <- torch_tensor(data_x, requires_grad = FALSE)
      data_v <- torch_tensor(data_v, requires_grad = FALSE)
      target <- torch_tensor(target, requires_grad = FALSE)

      if (args$cuda) {
        data_x <- data_x$cuda()
        data_v <- data_v$cuda()
        target <- target$cuda()
      }

      output <- model$forward(x = data_x, function_name = "autoencoder")
      enc <- output[[1]]
      pred <- output[[2]]

      embed <- enc$data$cpu()[,1,,drop=FALSE]
      final_embed[index] <- embed
    }

    final_embed <- final_embed$numpy()
    print(paste("    final_embed.shape=", dim(final_embed)))

    random_state <- sample(1:1234, 1)
    print(paste("random_state=", random_state))
    random_state_list <- c(random_state_list, random_state)
    kmeans <- kmeans(final_embed, centers = args$K_clusters, nstart = 25, iter.max = 1000)
    final_embed <- torch_tensor(final_embed)
    data_train$rep <- final_embed

    # Always put the high-risk outcome in the beginning
    oldlabel <- kmeans$cluster - 1
    result <- change_label_from_highratio_to_lowratio(args, oldlabel, data_train)
    new_labels <- result[[1]]
    order_c_map <- result[[2]]
    number_reassign <- sum(new_labels != data_train$C)
    print(paste("number_reassign=", number_reassign))
    number_reassign_list <- c(number_reassign_list, number_reassign)
    data_train$C <- new_labels

    data_train$n_cat <- args$K_clusters
    data_train$M <- torch_zeros(args$n_hidden_fea, args$K_clusters)
    print("***************************************")
    print(paste("data_train.M[0,:]=", data_train$M[1,]))
    update_M(data_train)
    print(paste("data_train.M[0,:]=", data_train$M[1,]))

    print(paste("    data_train.M.shape=", dim(data_train$M)))
    print(paste("    data_train.C.shape=", dim(data_train$C)))
    print(paste("    data_train.rep.shape=", dim(data_train$rep)))
    print("4. init train *.M, *.C, *.rep done!")

    # Update pseudo-label for test data
    update_testset_R_C_M_K(args, model, data_test, dataloader_test, data_train)

    # Use the kmeans label
    test_final_embed <- data_test$rep
    test_final_embed <- test_final_embed$numpy()
    test_cluster_old_labels <- predict(kmeans, newdata = test_final_embed)
    test_list_c <- test_cluster_old_labels - 1
    test_new_list_c <- sapply(test_list_c, function(x) order_c_map[as.character(x)])
    data_test$C <- torch_tensor(test_new_list_c)

    # Classification and regression
    list_train_AE_loss <- c()
    list_train_classifier_loss <- c()
    list_train_outcome_loss <- c()
    list_train_p_value_loss <- c()
    list_train_p_value_max <- c()
    list_train_p_value_min <- c()
    list_outcome_likelihood <- c()

    for (epoch in seq_len(args$epoch_in_iter)) {
      print(paste("epoch=", epoch))
      print("---------------------------------------------------------------------------------")
      print(paste("iter_i = ", iter_i, ", epoch=", epoch))

      total <- 0
      correct <- 0
      error_AE <- c()
      error_classifier <- c()
      error_outcome <- c()
      error_p_value <- c()
      error_outcome_likelihood <- c()
      outcome_true_y <- c()
      outcome_pred_prob <- c()
      print("-----------------")
      model$train()
      dataloader_train$reset()
      while (dataloader_train$has_next()) {
        #### <-------------
        #batch <- dataloader_train$next()
        index <- batch[[1]]
        batch_xvy <- batch[[2]]
        batch_c <- batch[[3]]

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

        # x, function, demov, mask_BoolTensor
        output <- model$forward(x = data_x, function_name = "outcome_logistic_regression", demov = data_v)
        encoded_x <- output[[1]]
        decoded_x <- output[[2]]
        output_c_no_activate <- output[[3]]
        output_outcome <- output[[4]]

        #############################
        # For calculate p-value, mask k1, or, mask k1 and k2 together, to calculate G
        list_k_in_c <- seq_len(args$K_clusters)
        k1 <- sample(list_k_in_c, 1)
        k2 <- sample(list_k_in_c, 1)
        while (k2 == k1) {
          k2 <- sample(list_k_in_c, 1)
        }

        list_mask_k1 <- rep(0, args$K_clusters)
        list_mask_k1[k1] <- 1

        list_mask_k1k2 <- rep(0, args$K_clusters)
        list_mask_k1k2[c(k1, k2)] <- 1

        mask_k1_tensor <- torch_tensor(list_mask_k1, dtype = torch_bool())
        mask_k1k2_tensor <- torch_tensor(list_mask_k1k2, dtype = torch_bool())

        output_mask_k1 <- model$forward(x = data_x, function_name = "outcome_logistic_regression", demov = data_v, mask_BoolTensor = mask_k1_tensor)
        output_mask_k1k2 <- model$forward(x = data_x, function_name = "outcome_logistic_regression", demov = data_v, mask_BoolTensor = mask_k1k2_tensor)

        encoded_x_mask_k1 <- output_mask_k1[[1]]
        decoded_x_mask_k1 <- output_mask_k1[[2]]
        output_c_no_activate_mask_k1 <- output_mask_k1[[3]]
        output_outcome_mask_k1 <- output_mask_k1[[4]]

        encoded_x_mask_k1k2 <- output_mask_k1k2[[1]]
        decoded_x_mask_k1k2 <- output_mask_k1k2[[2]]
        output_c_no_activate_mask_k1k2 <- output_mask_k1k2[[3]]
        output_outcome_mask_k1k2 <- output_mask_k1k2[[4]]
        ###############################

        optimizer$zero_grad()
        loss_classifier <- criterion_CrossEntropy(output_c_no_activate, batch_c)
        loss_AE <- criterion_MSE(data_x, decoded_x)
        loss_outcome <- criterion_BCE(output_outcome, target$float())
        loss_outcome_mask_k1 <- criterion_BCE(output_outcome_mask_k1, target$float())
        loss_outcome_mask_k1k2 <- criterion_BCE(output_outcome_mask_k1k2, target$float())
        loss_G <- 2 * (loss_outcome_mask_k1k2 - loss_outcome_mask_k1)
        loss_p_value <- 3.841 - loss_G

        loss <- args$lambda_AE * loss_AE +
          args$lambda_classifier * loss_classifier +
          args$lambda_outcome * loss_outcome +
          args$lambda_p_value * loss_p_value
        loss$backward()
        optimizer$step()

        error_AE <- c(error_AE, as.numeric(loss_AE$item()))
        error_classifier <- c(error_classifier, as.numeric(loss_classifier$item()))
        error_outcome <- c(error_outcome, as.numeric(loss_outcome$item()))
        error_p_value <- c(error_p_value, as.numeric(loss_p_value$item()))
        error_outcome_likelihood <- c(error_outcome_likelihood, as.numeric(loss_outcome$item()))

        predicted <- torch_max(output_c_no_activate$data(), 1)$indices
        data_train$pred_C[index] <- predicted$cpu()
        total <- total + batch_c$size(0)
        correct <- correct + sum(as.numeric(predicted == batch_c))

        outcome_true_y <- c(outcome_true_y, as.numeric(target$data()))
        outcome_pred_prob <- c(outcome_pred_prob, as.numeric(output_outcome$data()))
      }

      train_outcome_auc_score <- auc(outcome_true_y, outcome_pred_prob)
      print(paste("total=", total))
      classifier_c_accuracy <- correct / total

      train_AE_loss <- mean(error_AE)
      train_classifier_loss <- mean(error_classifier)
      train_outcome_loss <- mean(error_outcome)
      train_p_value_loss <- mean(error_p_value)
      train_outcome_likeilhood <- mean(error_outcome_likelihood)

      list_train_AE_loss <- c(list_train_AE_loss, train_AE_loss)
      list_train_classifier_loss <- c(list_train_classifier_loss, train_classifier_loss)
      list_train_outcome_loss <- c(list_train_outcome_loss, train_outcome_loss)

      list_train_p_value_max <- c(list_train_p_value_max, max(error_p_value))
      list_train_p_value_min <- c(list_train_p_value_min, min(error_p_value))
      list_train_p_value_loss <- c(list_train_p_value_loss, train_p_value_loss)

      list_outcome_likelihood <- c(list_outcome_likelihood, train_outcome_likeilhood)

      test_results <- func_analysis_test_error_D0406(args, model, data_test, dataloader_test)
      test_AE_loss <- test_results$test_AE_loss
      test_classifier_c_accuracy <- test_results$test_classifier_c_accuracy
      test_outcome_likelihood <- test_results$test_outcome_likelihood
      test_outcome_auc_score <- test_results$outcome_auc_score

      iter_train_negloglikeli_list <- c(iter_train_negloglikeli_list, train_outcome_likeilhood)
      iter_test_negloglikeli_list <- c(iter_test_negloglikeli_list, test_outcome_likelihood)
      iter_train_classifier_acc_list <- c(iter_train_classifier_acc_list, classifier_c_accuracy)
      iter_test_classifier_acc_list <- c(iter_test_classifier_acc_list, test_classifier_c_accuracy)
      iter_train_AE_list <- c(iter_train_AE_list, train_AE_loss)
      iter_test_AE_list <- c(iter_test_AE_list, test_AE_loss)
      iter_train_auc_list <- c(iter_train_auc_list, train_outcome_auc_score)
      iter_test_auc_list <- c(iter_test_auc_list, test_outcome_auc_score)

      print(sprintf("epoch %2d: train AE loss= %.4e, c acc= %.4e, outcome nll= %.4e, outcome_auc_score= %.4e, classifier loss= %.4e, outcome loss= %.4e, p_value loss= %.4e,",
                    epoch, train_AE_loss, classifier_c_accuracy, train_outcome_likeilhood, train_outcome_auc_score, train_classifier_loss, train_outcome_loss, train_p_value_loss))
      print(sprintf("        : test  AE loss= %.4e, c acc= %.4e, outcome nll= %.4e, outcome_auc_score= %.4e",
                    test_AE_loss, test_classifier_c_accuracy, test_outcome_likelihood, test_outcome_auc_score))

      #dict_outcome_ratio
      dict_p_value <- analysis_p_value_related(data_train, args$K_clusters, 1)
      dict_p_value_list <- unlist(dict_p_value)
      flag_morethan_0p05 <- any(dict_p_value_list > 0.05)

      if (test_outcome_likelihood < min_test_negloglikeli_record && !flag_morethan_0p05) {
        print(paste("save model here! iter_i=", iter_i, ", epoch=", epoch))
        min_test_negloglikeli_record <- test_outcome_likelihood
        torch_save(model$state_dict(), file.path(part2_foldername, 'model_iter.pt'))
        print("    Saving model")

        saveRDS(data_train, file = file.path(part2_foldername, 'data_train_iter.rds'))
        print("    save data_train")

        saved_iter_list <- c(saved_iter_list, iter_i)
        saved_iter <- iter_i
      }
    }
  }

  print(paste("number_reassign_list=", number_reassign_list))

  min_iter_test_negloglikel <- min(iter_test_negloglikeli_list)
  index_min_iter_test_negloglikeli <- which.min(iter_test_negloglikeli_list)
  number_reassign_list_tosee <- number_reassign_list[seq_len(index_min_iter_test_negloglikeli)]
  train_negloglikeli_tosee <- iter_train_negloglikeli_list[index_min_iter_test_negloglikeli]
  classifier_acc_tosee <- iter_train_classifier_acc_list[index_min_iter_test_negloglikeli]

  print(paste("random_state_list=", random_state_list))
  print(paste("saved_iter_list=", saved_iter_list))
  print(paste("saved_iter = ", saved_iter))
}
