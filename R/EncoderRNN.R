#' EncoderRNN Neural Network Module
#'
#' An RNN-based encoder module using Long Short-Term Memory (LSTM) architecture, implemented with the \code{torch} package.
#' This module is designed to process sequential data and is commonly used in tasks like sequence-to-sequence modeling, language modeling, and time series forecasting.
#'
#' @section Usage:
#' \preformatted{
#' EncoderRNN <- nn_module(
#'   "EncoderRNN",
#'
#'   initialize = function(input_size, nhidden, nlayers, dropout) {
#'     ...
#'   },
#'
#'   forward = function(x) {
#'     ...
#'   }
#' )
#' }
#'
#' @section Methods:
#' \describe{
#'   \item{\code{initialize(input_size, nhidden, nlayers, dropout)}}{
#'     Initializes the EncoderRNN module.
#'
#'     \itemize{
#'       \item{\code{input_size}: A numeric value representing the number of features in the input data.}
#'       \item{\code{nhidden}: A numeric value representing the number of hidden units in each LSTM layer.}
#'       \item{\code{nlayers}: A numeric value indicating the number of LSTM layers.}
#'       \item{\code{dropout}: A numeric value between 0 and 1 indicating the dropout probability to be applied between LSTM layers.}
#'     }
#'   }
#'
#'   \item{\code{forward(x)}}{
#'     Executes the forward pass of the EncoderRNN module.
#'
#'     \itemize{
#'       \item{\code{x}: A tensor of shape (batch_size, sequence_length, input_size) representing the input data.}
#'     }
#'
#'     \value{
#'     A list containing:
#'       \itemize{
#'         \item{\code{output}: The output tensor from the LSTM, processed and flipped.}
#'         \item{\code{list(hn, cn)}: A list containing the hidden state (\code{hn}) and cell state (\code{cn}) from the LSTM.}
#'         \item{\code{newinput}: A tensor representing the input data with an additional zero tensor at the beginning.}
#'       }
#'     }
#'   }
#' }
#' @import torch



EncoderRNN <- function(input_size, nhidden, nlayers, dropout) {
  EncoderRNN_out <- torch::nn_module(
  "EncoderRNN",

  initialize = function(input_size, nhidden, nlayers, dropout) {
    self$nhidden <- nhidden
    self$feasize <- input_size
    self$nlayers <- nlayers
    self$dropout <- dropout

    # Define the LSTM layer
    self$lstm <- nn_lstm(
      input_size = input_size, #self$feasize,
      hidden_size = nhidden, #self$nhidden,
      num_layers = nlayers, #self$nlayers,
      dropout = dropout, #self$dropout,
      batch_first = TRUE
    )

  },

  forward = function(x) {
    batch_size <- x$size(1)

    # Forward pass through LSTM
    out_state <- self$lstm(x)
    output <- out_state[[1]]
    state <- out_state[[2]]
    hn <- state[[1]]
    cn <- state[[2]]

    # Process the output
    output <- torch_flip(output, dims = c(1))
    newinput <- torch_flip(x, dims = c(1))

    # Add a zero tensor
    zeros <- torch_zeros(c(batch_size, 1, x$size(3)))
    newinput <- torch_cat(list(zeros, newinput), 2)

    # Return the processed output and new input
    list(output, list(hn, cn), newinput)
  }

  )
  return(EncoderRNN_out)
  }
