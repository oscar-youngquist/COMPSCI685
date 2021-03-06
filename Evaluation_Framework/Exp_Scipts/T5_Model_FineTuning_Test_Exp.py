import os
from os.path import join, exists
import sys
import logging
from pathlib import Path
from argparse import ArgumentParser

# update the python path to include the parent directory
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
print(parentdir)
sys.path.append(parentdir)

# load the experiment runner module
from ExperimentRunner import ExperimentRunner
from utils.UserDataReader import UserDataReader

##############
### TODO #####
##############
# import your own models here
from Models.T5_Model import T5_Base

##############
### TODO #####
##############
# add your model names here
model_name = "T5_test"


###
#    Helper function used to replaces sys.excepthook to log exceptions to
#        the same log file as everything else
###
def exceptionhook(self, *args):
    logging.getLogger().error("Exception: ", exc_info=args)


# get the current working directory
cwd = Path(__file__).parent.parent

# print(join(cwd, "Logs/"))

# make the overall Log folder if it does not exist
if not exists(join(cwd, "Logs/")):
    os.makedirs(join(cwd, "Logs/"))

# define the file for the logging object

##############
### TODO #####
##############
# TODO: add your own log file name
log_file = join(cwd, "Logs/", "t5_test.log")

# set up logger
root = logging.getLogger()
root.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
root.addHandler(console_handler)

file_handler = logging.FileHandler(log_file, mode="w", encoding=None, delay=False)
file_handler.setLevel(logging.DEBUG)
root.addHandler(file_handler)

# add the exception logging behavior to the logger
# sys.excepthook = exceptionhook


##############
### TODO #####
##############
# variables for experiment
num_examples = 5  # number of user-summaries used as examples
num_test = 3  # number of user-summaries used for testing
shared_docs_path = os.path.join(Path(__file__).parent.parent.parent, "SubSumE_Data")
data_path = os.path.join(shared_docs_path, "processed_state_sentences.csv")  # path to the processed sentences csv
users_path = os.path.join(shared_docs_path, "Test/")  # path to the misc. shared data (might not be needed anymore)
min_range = 0  # these min/max values are left-over from multi-processing experiments in which we would create n SuDocu models and then have each process (# of total user-summary instances)/n users. Thus we needed a min/max for the files to be read into memory by each model
max_range = 137
num_trials = 1  # number of times to evaluate all the examples
exp_folder = ""  # results folder for this experimental run, only used if running a) more than one model or b) the same model more than once
#     model_name folder is added as a sub-folder to this one

# create an instance of the experiment runner class
# num_examples, num_test, users_path, data_path, min_range, max_index
exp_runner = ExperimentRunner(num_examples, num_test, users_path, data_path, min_range, max_range, shared_docs_path)
os.environ["WANDB_SILENT"] = "true"
# create the model(s) you are going to evaluate
#     data_path, shared_docs_path, num_examples
# if finetune:
#    pegasus_model = T5_Base(data_path, shared_docs_path, num_examples, finetune=True)
# else:
#    pegasus_model = T5_Base(data_path, shared_docs_path, num_examples)

# perform the actual experiment
#   Could loop over several models/params. In Models, can
#   define extra params for stuff as needed and then instantiate
#   as needed.
#
#     model, num_trials, save_results (print results to log file), model_name, exp_folder=None, multi_processing=True // artifcact, huggingface API does not allow for multi-processing

T5_Base_model = T5_Base(data_path, shared_docs_path, num_examples, 5e-8, 3, 20)
exp_runner.get_model_analysis_final(T5_Base_model, True, model_name, multi_processing=False)
