#' Analyze P-Value and Outcome Ratios for Clusters
#'
#' This function performs an analysis of p-values and outcome ratios related to clusters in a dataset. It calculates the ratio of positive outcomes within each cluster and computes p-values for the association between clusters and the outcome variable.
#'
#' @param data_train A list representing the training data. This list must include the following components:
#' \itemize{
#'   \item \code{C}: An integer vector of cluster assignments for each sample.
#'   \item \code{data_v}: A matrix or data frame of additional covariates.
#'   \item \code{data_y}: A binary response vector with values 0 or 1, representing the outcome variable.
#' }
#' @param num_clusters An integer specifying the number of clusters in the data.
#' @param if_check A logical value indicating whether to print intermediate steps for debugging purposes. Default is \code{FALSE}.
#'
#' @return A list containing the following components:
#' \itemize{
#'   \item \code{dict_outcome_ratio}: A numeric vector representing the ratio of positive outcomes (where \code{data_y} equals 1) within each cluster.
#'   \item \code{dict_p_value}: A list of p-values, with each element corresponding to a pair of clusters. The p-values are calculated using a likelihood ratio test for the association between the clusters and the outcome variable.
#' }
#'
#' @details
#' The \code{analysis_p_value_related} function performs the following steps:
#' \enumerate{
#'   \item Converts the cluster assignments into a one-hot encoded matrix and counts the occurrences of each cluster.
#'   \item Calculates the ratio of positive outcomes within each cluster.
#'   \item Combines the cluster one-hot encodings with additional covariates to create a matrix of predictor variables.
#'   \item Iteratively removes one or two clusters from the predictor matrix and calculates the p-value for the remaining clusters using the \code{\link{p_value_calculate}} function.
#' }

p_value_calculate <- function(X, y, is_intercept, X_null = NULL) {
  full_model <- glm(y ~ X, family = binomial())
  alt_log_likelihood <- logLik(full_model)

  if (is_intercept) {
    null_prob <- mean(y)
    null_log_likelihood <- sum(dbinom(y, size = 1, prob = null_prob, log = TRUE))
    df <- 1
    G <- 2 * (alt_log_likelihood - null_log_likelihood)
    p_value <- pchisq(G, df, lower.tail = FALSE)
  } else {
    null_model <- glm(y ~ X_null, family = binomial())
    null_log_likelihood <- logLik(null_model)

    df <- ncol(X) - ncol(X_null)
    G <- 2 * (alt_log_likelihood - null_log_likelihood)
    p_value <- pchisq(G, df, lower.tail = FALSE)
  }
  return(p_value)
}

# Define the analysis_p_value_related function
analysis_p_value_related <- function(data_train, num_clusters, if_check = FALSE) {
  data_C <- data_train$C
  data_v <- data_train$data_v
  data_y <- data_train$data_y

  list_c <- as.integer(data_C)
  list_onehot <- matrix(0, nrow = length(list_c), ncol = num_clusters)
  dict_c_count <- integer(num_clusters)
  dict_outcome_in_c_count <- integer(num_clusters)

  for (i in seq_along(list_c)) {
    list_onehot[i, list_c[i] + 1] <- 1
    dict_c_count[list_c[i] + 1] <- dict_c_count[list_c[i] + 1] + 1
    if (data_y[i] == 1) {
      dict_outcome_in_c_count[list_c[i] + 1] <- dict_outcome_in_c_count[list_c[i] + 1] + 1
    }
  }

  if (if_check) {
    cat("--------\n")
    cat("num_clusters=", num_clusters, "\n")
    cat("\n")
    cat("list_c[0]=", list_c[1], "\n")
    cat("list_onehot[0,]=", list_onehot[1,], "\n")
    cat("\n")
    cat("list_c[1]=", list_c[2], "\n")
    cat("list_onehot[1,]=", list_onehot[2,], "\n")
    cat("--------\n")
  }
  cat("dict_c_count=", dict_c_count, "\n")
  cat("dict_outcome_in_c_count=", dict_outcome_in_c_count, "\n")

  dict_outcome_ratio <- dict_outcome_in_c_count / dict_c_count
  cat("dict_outcome_ratio=", dict_outcome_ratio, "\n")

  var_c <- list_onehot
  var_v <- data_v
  depend_y <- data_y

  var_cpv <- cbind(var_c, var_v)
  if (if_check) {
    cat("var_c.shape=", dim(var_c), ", var_v.shape=", dim(var_v), ", depend_y.shape=", length(depend_y), "\n")
    cat("var_cpv.shape=", dim(var_cpv), "\n")
  }

  cat("analysis done!\n")
  dict_p_value <- list()
  for (k1 in seq_len(num_clusters) - 1) {
    X_remove_k1 <- var_cpv
    slices_k1 <- setdiff(seq_len(ncol(var_cpv)), k1 + 1)
    X_remove_k1 <- X_remove_k1[, slices_k1, drop = FALSE]

    for (k2 in seq(k1 + 2, num_clusters)) {
      X_remove_k1k2 <- var_cpv
      slices_k1k2 <- setdiff(seq_len(ncol(var_cpv)), c(k1 + 1, k2))
      X_remove_k1k2 <- X_remove_k1k2[, slices_k1k2, drop = FALSE]

      cat("---------\n")
      cat("k1=", k1, ", k2=", k2 - 1, "\n")
      cat("slices_k1=", slices_k1, "\n")
      cat("slices_k1k2=", slices_k1k2, "\n")
      p_value_k1k2 <- p_value_calculate(X_remove_k1, depend_y, 0, X_remove_k1k2)
      dict_p_value[[paste(k1, k2 - 1, sep = ",")]] <- p_value_k1k2
    }
  }
  cat("dict_p_value=", dict_p_value, "\n")
  return(list(dict_outcome_ratio = dict_outcome_ratio, dict_p_value = dict_p_value))
}
