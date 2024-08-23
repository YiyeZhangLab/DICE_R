#' model_2 Neural Network Module
#'
#' A neural network module implementing a multi-functional model with encoder-decoder architecture using the \code{torch} package.
#' The model supports autoencoding, representation learning, classification, and logistic regression with optional demographic feature input.
#'
#' @section Usage:
#' \preformatted{
#' model_2 <- nn_module(
#'   "model_2",
#'
#'   initialize = function(input_size, nhidden, nlayers, dropout, n_clusters, n_dummy_demov_fea, para_cuda) {
#'     ...
#'   },
#'
#'   init_weights = function() {
#'     ...
#'   },
#'
#'   forward = function(x, function_name, demov = NULL, mask_BoolTensor = NULL) {
#'     ...
#'   }
#' )
#' }
#'
#' @section Methods:
#' \describe{
#'   \item{\code{initialize(input_size, nhidden, nlayers, dropout, n_clusters, n_dummy_demov_fea, para_cuda)}}{
#'     Initializes the model_2 module.
#'
#'     \itemize{
#'       \item{\code{input_size}: A numeric value representing the number of features in the input data.}
#'       \item{\code{nhidden}: A numeric value representing the number of hidden units in each LSTM layer.}
#'       \item{\code{nlayers}: A numeric value indicating the number of LSTM layers.}
#'       \item{\code{dropout}: A numeric value between 0 and 1 indicating the dropout probability to be applied between LSTM layers.}
#'       \item{\code{n_clusters}: A numeric value representing the number of clusters (or classes) for the classifier.}
#'       \item{\code{n_dummy_demov_fea}: A numeric value indicating the number of dummy demographic features used in the logistic regression.}
#'       \item{\code{para_cuda}: A logical value indicating whether CUDA (GPU acceleration) should be used.}
#'     }
#'   }
#'
#'   \item{\code{init_weights()}}{
#'     Initializes the weights of the linear layers in the model with uniform random values and biases set to zero.
#'   }
#'
#'   \item{\code{forward(x, function_name, demov = NULL, mask_BoolTensor = NULL)}}{
#'     Executes the forward pass of the model based on the specified function.
#'
#'     \itemize{
#'       \item{\code{x}: A tensor of shape (batch_size, sequence_length, input_size) representing the input data.}
#'       \item{\code{function_name}: A character string indicating the type of operation to perform, one of \code{"autoencoder"}, \code{"get_representation"}, \code{"classifier"}, or \code{"outcome_logistic_regression"}.}
#'       \item{\code{demov}: (Optional) A tensor representing demographic feature data, used in logistic regression. Default is \code{NULL}.}
#'       \item{\code{mask_BoolTensor}: (Optional) A boolean tensor used to mask certain values during the logistic regression operation. Default is \code{NULL}.}
#'     }
#'
#'     \value{
#'       A list containing outputs depending on the chosen \code{function_name}:
#'       \itemize{
#'         \item{\code{"autoencoder"}: A list with encoded and decoded tensors.}
#'         \item{\code{"get_representation"}: A tensor representing the encoded input.}
#'         \item{\code{"classifier"}: A list with the encoded tensor and class probabilities.}
#'         \item{\code{"outcome_logistic_regression"}: A list containing the encoded tensor, decoded tensor, unactivated class scores, and logistic regression output.}
#'       }
#'     }
#'   }
#' }
#' @import torch

model_2 <- function(input_size, nhidden, nlayers, dropout, n_clusters, n_dummy_demov_fea, para_cuda) {
  model_2_out <- torch::nn_module(
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
          output_c <- output_c$masked_fill(mask = mask_BoolTensor, value = torch_tensor(0.0))
        }

        output_from_c <- self$linear_regression_c(output_c)
        output_from_v <- self$linear_regression_demov(demov)
        output_cpv <- output_from_c + output_from_v
        output_outcome <- self$activation_regression(output_cpv)

        return(list(encoded_x, decoded_x, output_c_no_activate, output_outcome))
      } else {
        print("No corresponding function, check the function you want for model_2")
        return("Wrong!")
      }
    }
  )
  return(model_2_out)
}
