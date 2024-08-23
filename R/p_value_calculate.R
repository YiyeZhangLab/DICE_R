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
