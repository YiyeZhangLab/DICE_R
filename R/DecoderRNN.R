#' DecoderRNN Neural Network Module
#'
#' A neural network module implementing a decoder using an LSTM architecture, based on the \code{torch} package. This module is typically used in sequence-to-sequence tasks during the decoding phase.
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
##' @section Methods:
#' \describe{
#'   \item{\code{initialize(input_size, nhidden, nlayers, dropout)}}{
#'     Initializes the \code{DecoderRNN} module.
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
#'     Executes the forward pass of the \code{DecoderRNN} module.
#'
#'     \itemize{
#'       \item{\code{x}: A tensor of shape \code{(batch_size, sequence_length, input_size)} representing the input data.}
#'       \item{\code{h}: A list containing the hidden state (\code{hn}) and cell state (\code{cn}) from the LSTM, typically passed from the encoder or previous decoder step.}
#'     }
#'
#'     \value{
#'       A tensor representing the output sequence, flipped along the sequence dimension.
#'     }
#'   }
#' }
#'
#' @param input_size A numeric value representing the number of features in the input data.
#' @param nhidden A numeric value representing the number of hidden units in each LSTM layer.
#' @param nlayers A numeric value indicating the number of LSTM layers.
#' @param dropout A numeric value between 0 and 1 indicating the dropout probability to be applied between LSTM layers.
# @param x A tensor of shape \code{(batch_size, sequence_length, input_size)} representing the input data.
# @param h A list containing the hidden state (\code{hn}) and cell state (\code{cn}) from the LSTM, typically passed from the encoder or previous decoder step.
#'
#' @return A tensor representing the output sequence, flipped along the sequence dimension.
#'
#' @import torch
#' @export


DecoderRNN <- nn_module(
  "DecoderRNN",

  initialize = function(input_size, nhidden, nlayers, dropout) {
    self$nhidden <- nhidden
    self$feasize <- input_size
    self$nlayers <- nlayers
    self$dropout <- dropout

    # Define the LSTM layer
    self$lstm <- nn_lstm(
      input_size = self$feasize,
      hidden_size = self$nhidden,
      num_layers = self$nlayers,
      dropout = self$dropout,
      batch_first = TRUE
    )

    # Initialize weights directly
    #params <- self$lstm$parameters
    #for (p in params) {
    #  p$data$uniform_(-0.1, 0.1)
    #}
  },

  forward = function(x, h) {
    # Forward pass through LSTM
    out_state <- self$lstm(x, h)
    output <- out_state[[1]]
    state <- out_state[[2]]

    # Flip the output tensor
    fin <- torch_flip(output, dims = c(1))

    return(fin)
  }
)
