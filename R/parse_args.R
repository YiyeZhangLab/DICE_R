#' Parse Command-Line Arguments for PPD-Aware Clustering
#'
#' This function parses command-line arguments for a script that performs PPD-aware clustering, setting up various parameters necessary for training and evaluating a model.
#'
#' @return A list of parsed command-line arguments, each corresponding to a specific parameter required by the clustering model. This list is typically used to configure the model training and evaluation process.
#'
#' @details
#' The \code{parse_args} function uses the \code{argparser} package to define and parse a series of command-line arguments. These arguments are essential for configuring various aspects of the PPD-aware clustering process, including model architecture, training settings, and input/output paths.
#'
#' The following command-line arguments are supported:
#' \itemize{
#'   \item \code{--init_AE_epoch}: Integer. Number of epochs for representation initialization.
#'   \item \code{--n_hidden_fea}: Integer. Number of hidden units in the LSTM layers.
#'   \item \code{--output_path}: Character. Location where output results will be saved.
#'   \item \code{--input_path}: Character. Location of the input dataset.
#'   \item \code{--filename_train}: Character. Path to the training data file.
#'   \item \code{--filename_test}: Character. Path to the test data file.
#'   \item \code{--n_input_fea}: Integer. Number of original input features.
#'   \item \code{--n_dummy_demov_fea}: Integer. Number of dummy demographic features.
#'   \item \code{--lstm_layer}: Integer. Number of layers in the LSTM. Default is 1.
#'   \item \code{--lr}: Numeric. Learning rate for training. Default is 1e-4.
#'   \item \code{--lstm_dropout}: Numeric. Dropout rate for the LSTM layers. Default is 0.0.
#'   \item \code{--K_clusters}: Integer. Number of initial clusters.
#'   \item \code{--iter}: Integer. Maximum number of iterations for merging clusters. Default is 20.
#'   \item \code{--epoch_in_iter}: Integer. Number of epochs per iteration when merging clusters. Default is 1.
#'   \item \code{--seed}: Integer. Random seed for reproducibility. Default is 1111.
#'   \item \code{--cuda}: Integer. Flag indicating whether to use CUDA (GPU acceleration). Default is 1.
#'   \item \code{--lambda_AE}: Numeric. Regularization weight for the autoencoder loss in each iteration. Default is 1.0.
#'   \item \code{--lambda_classifier}: Numeric. Regularization weight for the classifier loss in each iteration. Default is 1.0.
#'   \item \code{--lambda_outcome}: Numeric. Regularization weight for the outcome loss in each iteration. Default is 1.0.
#'   \item \code{--lambda_p_value}: Numeric. Regularization weight for the p-value adjustment in each iteration. Default is 1.0.
#' }
#' @import argparser

parse_args <- function() {
  # Defines the function parse_args() which doesn't accept any parameters and is intended to be called directly from the command line or within a script to parse arguments.
  parser <- arg_parser('ppd-aware clustering')

  # Adding Arguments
  # Each add_argument() call adds a specific command-line option to the parser. Here is what each one does:
  parser <- add_argument(parser, '--init_AE_epoch', type = "integer", help = 'number of epoch for representation initialization'#, required = TRUE
  )
  parser <- add_argument(parser, '--n_hidden_fea', type = "integer", help = 'number of hidden size in LSTM'#, required = TRUE
  )
  parser <- add_argument(parser, '--output_path', type = "character", help = 'location of output path')
  parser <- add_argument(parser, '--input_path', type = "character", help = 'location of input dataset'#, required = TRUE
  )
  parser <- add_argument(parser, '--filename_train', type = "character", help = 'location of the data corpus'#, required = TRUE
  )
  parser <- add_argument(parser, '--filename_test', type = "character", help = 'location of the data corpus'#, required = TRUE
  )
  parser <- add_argument(parser, '--n_input_fea', type = "integer", help = 'number of original input feature size'#, required = TRUE
  )
  parser <- add_argument(parser, '--n_dummy_demov_fea', type = "integer", help = 'number of dummy demo feature size'#, required = TRUE
  )
  parser <- add_argument(parser, '--lstm_layer', type = "integer", help = 'number of hidden size in LSTM', default = 1)
  parser <- add_argument(parser, '--lr', type = "numeric", help = 'learning rate', default = 1e-4)
  parser <- add_argument(parser, '--lstm_dropout', type = "numeric", help = 'dropout in LSTM', default = 0.0)
  parser <- add_argument(parser, '--K_clusters', type = "integer", help = 'number of initial clusters'#, required = TRUE
  )
  parser <- add_argument(parser, '--iter', type = "integer", help = 'maximum of iterations in iteration merge clusters', default = 20)
  parser <- add_argument(parser, '--epoch_in_iter', type = "integer", help = 'maximum of iterations in iteration merge clusters', default = 1)
  parser <- add_argument(parser, '--seed', type = "integer", help = 'random seed', default = 1111)
  parser <- add_argument(parser, '--cuda', type = "integer", help = 'If use cuda', default = 1)
  parser <- add_argument(parser, '--lambda_AE', type = "numeric", help = 'lambda of AE in iteration', default = 1.0)
  parser <- add_argument(parser, '--lambda_classifier', type = "numeric", help = 'lambda_classifier of classifier in iteration', default = 1.0)
  parser <- add_argument(parser, '--lambda_outcome', type = "numeric", help = 'lambda of outcome in iteration', default = 1.0)
  parser <- add_argument(parser, '--lambda_p_value', type = "numeric", help = 'lambda of p value in iteration', default = 1.0)

  # Parse & display arguments
  # - parser.parse_args(): Parses the command-line arguments provided to the script. If any required arguments are missing or invalid, the script will automatically display an error and help information.
  # - print(vars(args)): Converts the args Namespace object to a dictionary and prints it. This shows all the arguments that have been set, either by defaults or by user input.
  # - return args: Returns the populated args Namespace. This object contains all the command-line arguments that were parsed. This is useful for the rest of the program to access and use these settings.
  args <- parse_args(parse)
  foo <- parse_args()
  print("parameters:")
  print(args)
  return(args)
}
