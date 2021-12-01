import os
from os.path import join, exists
from pathlib import Path
import sys
import logging

# update the python path to include the parent directory  
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
print(parentdir)
sys.path.append(parentdir)

# load the experiment runner module
from ExperimentRunner import ExperimentRunner

# import your own models here
from Models.RandomN_Model import RandomN

###
#    Helper function used to replaces sys.excepthook to log exceptions to 
#        the same log file as everything else
### 
def exceptionhook(self, *args):
    logging.getLogger().error("Exception: ", exc_info=args)

# get the current working directory 
cwd = Path(__file__).parent.parent

# make the overall Log folder if it does not exist
if not exists(join(cwd, "Logs/")):
    os.makedirs(join(cwd, "Logs/"))

# define the file for the logging object
# TODO: add your own log file name
log_file = join(cwd, "Logs/", "RandomN.log")

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

# variables for experiment
num_examples = 5                                                                                        # number of user-summaries used as examples
num_test = 3                                 
shared_docs_path = os.path.join(Path(__file__).parent.parent.parent, "SubSumE_Data")
data_path = os.path.join(shared_docs_path, "processed_state_sentences.csv")        # path to the processed sentences csv
users_path = os.path.join(shared_docs_path, "Train/")                               # path to the user-files used for this experiment                              # path to the misc. shared data (might not be needed anymore)
min_range = 0                                                                                           # these min/max values are left-over from multi-processing experiments in which we would create n SuDocu models and then have each process (# of total user-summary instances)/n users. Thus we needed a min/max for the files to be read into memory by each model
max_range = 138
num_trials = 1 # 1 trial is OK for LeadN because it's deterministic                                                                                    # number of times to evaluate all the examples
model_name = "RandomN"                                                                                  # name of the model used for this experiment (creates a folder with this name in the results directory)
exp_folder = ""                                                                                         # results folder for this experimental run, only used if running a) more than one model or b) the same model more than once
                                                                                                        #     model_name folder is added as a sub-folder to this one 

# create an instance of the experiment runner class
# num_examples, num_test, users_path, data_path, min_range, max_index
exp_runner = ExperimentRunner(num_examples, num_test, users_path, data_path, min_range, max_range)

# create the model(s) you are going to evaluate
#     data_path, shared_docs_path, num_examples, max_solvs=50, length_modifier=0.25
random_n_model = RandomN(data_path, shared_docs_path, num_examples)

# perform the actual experiment
#     model, num_trials, display_results, model_name, exp_folder=None, multi_processing=True
exp_runner.get_model_analysis_final(random_n_model, num_trials, True, model_name, multi_processing=False)
