#' Reassign Cluster Labels Based on Outcome Ratios
#'
#' This function reassigns cluster labels such that clusters with higher ratios of positive outcomes are assigned lower labels, and those with lower ratios are assigned higher labels. This relabeling can be useful for aligning cluster labels with the significance of outcomes.
#'
#' @param args A list of arguments, which should include the following component:
#' \itemize{
#'   \item \code{K_clusters}: An integer specifying the number of clusters.
#' }
#' @param oldlabel An integer vector of the original cluster labels assigned to each sample.
#' @param data_train A list representing the training data. This list must include the following components:
#' \itemize{
#'   \item \code{data_v}: A matrix or data frame of additional covariates.
#'   \item \code{data_y}: A binary response vector with values 0 or 1, representing the outcome variable.
#' }
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{new_list_c}: A tensor of the new cluster labels, with clusters relabeled according to the outcome ratios.
#'   \item \code{order_c_map}: A named vector that maps the old cluster labels to the new ones, where the names represent the original labels and the values represent the new labels.
#' }
#'
#' @details
#' The \code{change_label_from_highratio_to_lowratio} function performs the following steps:
#' \enumerate{
#'   \item Calculates the count of samples and the count of positive outcomes (where \code{data_y} equals 1) for each cluster.
#'   \item Computes the ratio of positive outcomes within each cluster.
#'   \item Orders the clusters by their outcome ratios in descending order.
#'   \item Reassigns cluster labels such that clusters with higher outcome ratios receive lower labels.
#'   \item Returns the new cluster labels and the mapping from old labels to new labels.
#' }
#' @import torch

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
