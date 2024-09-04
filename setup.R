#############################################################
#                     Writing an R package                  #
#############################################################

########################### Set-up ##########################

library(devtools)
library(roxygen2)
library(testthat)

packageVersion("devtools")

devtools::session_info()

####################### Create Package #######################

#create_package("/Users/sarahayton/Documents/Cornell/DICE+MuTATE/DICEr")

use_git()

# Add dependencies
usethis::use_package("torch")
usethis::use_package("argparser")
usethis::use_package("ggplot2")

#usethis::use_package("stats")
#usethis::use_package("binomial")
#usethis::use_package("dbinom")
#usethis::use_package("glm")
#usethis::use_package("logLik")
#usethis::use_package("pchisq")
#usethis::use_package("predict")
#usethis::use_package("setNames")

# Set license
usethis::use_mit_license()


####################### Write Functions ######################

use_r("EncoderRNN")
use_r("DecoderRNN")
use_r("model_2")
use_r("parse_args")
use_r("test_AE")
use_r("update_M")
use_r("update_testset_R_C_M_K") #batch <- dataloader_test$next()
use_r("p_value_calculate")
use_r("analysis_p_value_related")
use_r("analysis_p_value_related_onlyc")
use_r("func_analysis_test_error_D0406") #batch <- dataloader_train$next()
use_r("change_label_from_highratio_to_lowratio")
use_r("main") #dict_outcome_ratio, #batch <- dataloader_train$next()


load_all()
check()

EncoderRNN()

#############################################################
