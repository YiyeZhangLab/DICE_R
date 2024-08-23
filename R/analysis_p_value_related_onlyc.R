analysis_p_value_related_onlyc <- function(data_train, num_clusters, if_check = FALSE) {
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

  cat("analysis done!\n")
  dict_p_value <- list()
  for (k1 in seq_len(num_clusters) - 1) {
    X_remove_k1 <- var_c
    slices_k1 <- setdiff(seq_len(ncol(var_c)), k1 + 1)
    X_remove_k1 <- X_remove_k1[, slices_k1, drop = FALSE]

    for (k2 in seq(k1 + 2, num_clusters)) {
      X_remove_k1k2 <- var_c
      slices_k1k2 <- setdiff(seq_len(ncol(var_c)), c(k1 + 1, k2))
      X_remove_k1k2 <- X_remove_k1k2[, slices_k1k2, drop = FALSE]

      cat("---------\n")
      cat("k1=", k1, ", k2=", k2 - 1, "\n")
      cat("slices_k1=", slices_k1, "\n")
      cat("slices_k1k2=", slices_k1k2, "\n")
      p_value_k1k2 <- p_value_calculate(X_remove_k1, depend_y, 0, X_remove_k1k2)
      dict_p_value[[paste(k1, k2 - 1, sep = ",")]] <- p_value_k1k2
    }
  }
  cat("dict_p_value only c=", dict_p_value, "\n")
  cat("analysis done!\n")
}
