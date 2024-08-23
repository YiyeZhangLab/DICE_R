require(torch)
change_label_from_highratio_to_lowratio <- function(args, oldlabel, data_train) {
  data_v <- data_train$data_v
  data_y <- data_train$data_y

  list_c <- as.integer(oldlabel)
  dict_c_count <- integer(args$K_clusters)
  dict_outcome_in_c_count <- integer(args$K_clusters)

  for (i in seq_along(list_c)) {
    dict_c_count[list_c[i] + 1] <- dict_c_count[list_c[i] + 1] + 1
    if (data_y[i] == 1) {
      dict_outcome_in_c_count[list_c[i] + 1] <- dict_outcome_in_c_count[list_c[i] + 1] + 1
    }
  }

  dict_outcome_ratio <- dict_outcome_in_c_count / dict_c_count
  cat("Before change dict_outcome_ratio =", dict_outcome_ratio, "\n")

  sorted_indices <- order(dict_outcome_ratio, decreasing = TRUE)
  order_c_map <- setNames(seq_along(sorted_indices) - 1, sorted_indices - 1)
  cat("sorted_dict_outcome_ratio =", dict_outcome_ratio[sorted_indices], "\n")
  cat("order_c_map =", order_c_map, "\n")

  new_list_c <- sapply(list_c, function(x) order_c_map[as.character(x)])

  return(list(torch_tensor(new_list_c, dtype = torch_int()), order_c_map))
}
