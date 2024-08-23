#' DecoderRNN Neural Network Module
#'
#' An RNN-based decoder module using Long Short-Term Memory (LSTM) architecture, implemented with the \code{torch} package.
#' This module is designed to process sequential data during the decoding phase in tasks like sequence-to-sequence modeling.
#'
#' @section Usage:
#' \preformatted{
#' DecoderRNN <- nn_module(
#'   "DecoderRNN",
#'
#'   initialize = function(input_size, nhidden, nlayers, dropout) {
#'     ...
#'   },
#'
#'   forward = function(x, h) {
#'     ...
#'   }
#' )
#' }
#'
#' @section Methods:
#' \describe{
#'   \item{\code{initialize(input_size, nhidden, nlayers, dropout)}}{
#'     Initializes the DecoderRNN module.
#'
#'     \itemize{
#'       \item{\code{input_size}: A numeric value representing the number of features in the input data.}
#'       \item{\code{nhidden}: A numeric value representing the number of hidden units in each LSTM layer.}
#'       \item{\code{nlayers}: A numeric value indicating the number of LSTM layers.}
#'       \item{\code{dropout}: A numeric value between 0 and 1 indicating the dropout probability to be applied between LSTM layers.}
#'     }
#'   }
#'
#'   \item{\code{forward(x, h)}}{
#'     Executes the forward pass of the DecoderRNN module.
#'
#'     \itemize{
#'       \item{\code{x}: A tensor of shape (batch_size, sequence_length, input_size) representing the input data.}
#'       \item{\code{h}: A list containing the hidden state (\code{hn}) and cell state (\code{cn}) from the LSTM, typically passed from the encoder or previous decoder step.}
#'     }
#'
#'     \value{
#'       A tensor representing the output sequence, flipped along the sequence dimension.
#'     }
#'   }
#' }
#' @import torch
DecoderRNN <- function(input_size, nhidden, nlayers, dropout) {
  DecoderRNN_out <- torch::nn_module(
    "DecoderRNN",

    initialize = function(input_size, nhidden, nlayers, dropout) {
      self$nhidden <- nhidden
      self$feasize <- input_size
      self$nlayers <- nlayers
      self$dropout <- dropout

      # Define the LSTM layer
      self$lstm <- torch::nn_lstm(
        input_size = self$feasize,
        hidden_size = self$nhidden,
        num_layers = self$nlayers,
        dropout = self$dropout,
        batch_first = TRUE
      )
    },

    forward = function(x, h) {
      # Forward pass through LSTM
      out_state <- self$lstm(x, h)
      output <- out_state[[1]]
      state <- out_state[[2]]

      # Flip the output tensor
      fin <- torch::torch_flip(output, dims = c(1))

      return(fin)
    }
  )
  return(DecoderRNN_out)
}
