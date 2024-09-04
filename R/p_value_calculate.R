#' Calculate P-Value for Logistic Regression Model
#'
#' This function calculates the p-value for a logistic regression model by comparing the likelihoods of a full model and a null model.
#'
#' @param X A matrix of predictor variables for the full model.
#' @param y A binary response vector, with values of 0 or 1.
#' @param is_intercept A logical value indicating whether the null model should include only an intercept (i.e., no predictors). If \code{TRUE}, the null model includes only the intercept; if \code{FALSE}, the null model is specified by \code{X_null}.
#' @param X_null (Optional) A matrix of predictor variables for the null model. This argument is required if \code{is_intercept} is \code{FALSE}.
#'
#' @return A numeric value representing the p-value from the likelihood ratio test between the full and null models.
#'
#' @details
#' The \code{p_value_calculate} function performs the following steps:
#' \enumerate{
#'   \item Fits a full logistic regression model using \code{X} as the predictor matrix.
#'   \item Depending on the value of \code{is_intercept}, it either:
#'   \itemize{
#'     \item Calculates the null model's log-likelihood using only an intercept (if \code{is_intercept = TRUE}).
#'     \item Fits a null logistic regression model using \code{X_null} as the predictor matrix (if \code{is_intercept = FALSE}).
#'   }
#'   \item Computes the likelihood ratio statistic \code{G}, which is twice the difference in log-likelihoods between the full and null models.
#'   \item Calculates the p-value using the chi-squared distribution with degrees of freedom equal to the difference in the number of parameters between the full and null models.
#' }
#'
#' @examples
#' \dontrun{
#' # Example with only an intercept in the null model
#' X <- matrix(rnorm(100 * 5), ncol = 5)
#' y <- rbinom(100, 1, 0.5)
#' p_value <- p_value_calculate(X, y, is_intercept = TRUE)
#' print(p_value)
#'
#' # Example with a specified null model
#' X_null <- X[, 1:2]  # Using only the first two predictors in the null model
#' p_value <- p_value_calculate(X, y, is_intercept = FALSE, X_null = X_null)
#' print(p_value)
#' }

p_value_calculate <- function(X, y, is_intercept, X_null = NULL) {
  cat("X.shape=", dim(X), ", y.shape=", length(y), "\n")

  # Fit the full model
  full_model <- glm(y ~ X, family = binomial())
  alt_log_likelihood <- logLik(full_model)

  if (is_intercept) {
    # Calculate the null model with only intercept
    null_prob <- mean(y)
    null_log_likelihood <- sum(dbinom(y, size = 1, prob = null_prob, log = TRUE))
    df <- 1
    G <- 2 * (alt_log_likelihood - null_log_likelihood)
    p_value <- pchisq(G, df, lower.tail = FALSE)
  } else {
    # Fit the null model
    null_model <- glm(y ~ X_null, family = binomial())
    null_log_likelihood <- logLik(null_model)

    df <- ncol(X) - ncol(X_null)
    G <- 2 * (alt_log_likelihood - null_log_likelihood)
    p_value <- pchisq(G, df, lower.tail = FALSE)
  }
  return(p_value)
}
