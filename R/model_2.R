#' model_2 Neural Network Module
#'
#' A neural network module implementing a multi-functional model with
#' encoder-decoder architecture using the \code{torch} package.
#' The module is constructed using \code{nn_module()} and configured
#' via its \code{initialize()} method.
#'
#' @details
#' After initialization, the module can be used via
#' \code{model$forward(x, function_name, demov = NULL, mask_BoolTensor = NULL)}.
#'
#' Arguments:
#' \itemize{
#'   \item \code{x}: Input tensor.
#'   \item \code{function_name}: One of "autoencoder",
#'     "get_representation", "classifier",
#'     "outcome_logistic_regression".
#'   \item \code{demov}: Optional demographic tensor.
#'   \item \code{mask_BoolTensor}: Optional boolean mask tensor.
#' }
#'
#' @param input_size Integer. Number of features in the input data.
#' @param nhidden Integer. Number of hidden units in each LSTM layer.
#' @param nlayers Integer. Number of LSTM layers.
#' @param dropout Numeric. Dropout probability between LSTM layers (0–1).
#' @param n_clusters Integer. Number of clusters/classes.
#' @param n_dummy_demov_fea Integer. Number of demographic dummy features.
#' @param para_cuda Logical. Whether to use CUDA (GPU acceleration).
#'
#' The \code{forward()} method arguments:
#'   \code{"get_representation"}, \code{"classifier"},
#'   \code{"outcome_logistic_regression"}.
#'
#' @return
#' An \code{nn_module} object.
#'
#' @examples
#' \donttest{
#' if (requireNamespace("torch", quietly = TRUE)) {
#'   mod <- model_2(
#'     input_size = 10L, nhidden = 64L, nlayers = 2L, dropout = 0.1,
#'     n_clusters = 3L, n_dummy_demov_fea = 2L, para_cuda = FALSE
#'   )
#' }
#' }
#'
#' @usage model_2(input_size, nhidden, nlayers, dropout, n_clusters, n_dummy_demov_fea, para_cuda)
#' @import torch
#' @export


model_2 <- nn_module(
  "model_2",

  initialize = function(input_size, nhidden, nlayers, dropout, n_clusters, n_dummy_demov_fea, para_cuda) {
    self$nhidden <- nhidden
    self$input_size <- input_size
    self$nlayers <- nlayers
    self$dropout <- dropout
    self$n_clusters <- n_clusters
    self$n_dummy_demov_fea <- n_dummy_demov_fea
    self$para_cuda <- para_cuda

    self$encoder <- EncoderRNN(self$input_size, self$nhidden, self$nlayers, self$dropout)
    self$decoder <- DecoderRNN(self$input_size, self$nhidden, self$nlayers, self$dropout)

    self$linear_decoder_output <- nn_linear(self$nhidden, self$input_size)
    self$linear_classifier_c <- nn_linear(self$nhidden, self$n_clusters)
    self$activateion_classifier <- nn_softmax(dim = 1)
    self$linear_regression_c <- nn_linear(self$n_clusters, 1)
    self$linear_regression_demov <- nn_linear(self$n_dummy_demov_fea, 1)
    self$activation_regression <- nn_sigmoid()

    #self$init_weights()
  },

  init_weights = function() {
    self$linear_decoder_output$bias$data$fill_(0)
    self$linear_decoder_output$weight$data$uniform_(-0.1, 0.1)

    self$linear_classifier_c$bias$data$fill_(0)
    self$linear_classifier_c$weight$data$uniform_(-0.1, 0.1)

    self$linear_regression_c$bias$data$fill_(0)
    self$linear_regression_c$weight$data$uniform_(-0.1, 0.1)

    self$linear_regression_demov$bias$data$fill_(0)
    self$linear_regression_demov$weight$data$uniform_(-0.1, 0.1)
  },

  forward = function(x, function_name, demov = NULL, mask_BoolTensor = NULL) {
    if (function_name == "autoencoder") {
      result <- self$encoder(x)
      encoded_x <- result[[1]]
      state <- result[[2]]
      newinput <- result[[3]]
      decoded_x <- self$decoder(newinput, state)
      decoded_x <- self$linear_decoder_output(decoded_x)
      return(list(encoded_x, decoded_x))
    } else if (function_name == "get_representation") {
      result <- self$encoder(x)
      encoded_x <- result[[1]]
      return(encoded_x)
    } else if (function_name == "classifier") {
      result <- self$encoder(x)
      encoded_x <- result[[1]]
      output <- self$linear_classifier_c(encoded_x)
      output <- self$activateion_classifier(output)
      return(list(encoded_x, output))
    } else if (function_name == "outcome_logistic_regression") {
      result <- self$encoder(x)
      encoded_x <- result[[1]]
      state <- result[[2]]
      newinput <- result[[3]]
      decoded_x <- self$decoder(newinput, state)
      decoded_x <- self$linear_decoder_output(decoded_x)

      encoded_x <- encoded_x[,1,,drop=FALSE]
      output_c_no_activate <- self$linear_classifier_c(encoded_x)
      output_c <- self$activateion_classifier(output_c_no_activate)

      if (!is.null(mask_BoolTensor)) {
        if (self$para_cuda) {
          mask_BoolTensor <- mask_BoolTensor$cuda()
        }
        else {
          output_c <- output_c$masked_fill(mask = mask_BoolTensor, value = (0.0))
        }
      }

      output_from_c <- self$linear_regression_c(output_c)
      #demov <- demov$view(c(-1, 4))
      #demov <- demov$squeeze(dim = 1)

      output_from_v <- self$linear_regression_demov(demov)
      output_cpv <- output_from_c + output_from_v
      output_outcome <- self$activation_regression(output_cpv)

      return(list(encoded_x, decoded_x, output_c_no_activate, output_outcome))
    } else {
      message("No corresponding function, check the function you want for model_2")
      return("Wrong!")
    }
  }
)
