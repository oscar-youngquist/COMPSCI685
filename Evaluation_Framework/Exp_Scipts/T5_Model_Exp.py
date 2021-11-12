import os
from os.path import join, exists
import sys
import logging
from pathlib import Path
from argparse import ArgumentParser, Namespace
import wandb

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

parser = ArgumentParser(description="Experiment runner for t5 base and fine-tuned t5")
parser.add_argument('--use_wandb', action='store_true') # Use weights and biases to track hyperparameters and training runs
parser.add_argument('--wandb_silent', action='store_true') # Wandb silent mode, does not output any text about project run/syncing
parser.add_argument('--verbose', action='store_false') # Print every predicted example (on by default, for long runs it should be turned off to clean up output)
parser.add_argument('--finetune', action='store_true') # Fine tune model (as opposed to frozen model)
parser.add_argument('--data_aug', action='store_true') # Use data augmentation (UDA)
parser.add_argument('--num_aug', type=int, default=10) # Number of augmented examples per training example (max of 10 currently)
parser.add_argument('--gamma', type=float, default=1.0) # Data augmentation: coefficient for multiplying consistency loss (UDA uses 1)
parser.add_argument('--num_trials', type=int, default=10) # Number of complete prediction runs through all user examples (use 1 for hyperparameter tuning)
parser.add_argument('--num_examples', type=int, default=5) # Number of user-summaries used as examples
parser.add_argument('--num_test', type=int, default=3) # Number of user-summaries used for testing
parser.add_argument('--max_range', type=int, default=138) # Number of users, max of 138 (use 20 for hyperparameter tuning)
args = parser.parse_args()

config = Namespace()
config.use_wandb = args.use_wandb
config.verbose = args.verbose
config.finetune = args.finetune
config.data_aug = args.data_aug
config.num_aug = args.num_aug
config.gamma = args.gamma

# experiment configuration
config.num_examples = args.num_examples     # number of user-summaries used as examples
config.num_test = args.num_test     # number of user-summaries used for testing
config.shared_docs_path = os.path.join(Path(__file__).parent.parent.parent, "SubSumE_Data") # path to shared documents in dataset
config.aug_path = os.path.join(Path(__file__).parent.parent.parent, *["backtranslation", "paraphrases"]) # path to augmented sentences (backtranslation)
config.data_path = os.path.join(config.shared_docs_path, "processed_state_sentences.csv")      # path to the processed sentences csv
config.users_path = os.path.join(config.shared_docs_path, "train")     # path to the misc. shared data (might not be needed anymore)
config.min_range = 0    # these min/max values are left-over from multi-processing experiments in which we would create n SuDocu models and then have each process (# of total user-summary instances)/n users. Thus we needed a min/max for the files to be read into memory by each model
config.max_range = args.max_range
config.num_trials = args.num_trials     # number of times to evaluate all the examples
config.exp_folder = ""      # results folder for this experimental run, only used if running a) more than one model or b) the same model more than once
                                # model_name folder is added as a sub-folder to this one

# add your model names here
config.model_name = "T5_finetuned_test" if config.finetune else "T5_test"

if args.wandb_silent:
    os.environ["WANDB_SILENT"] = "true"

wandb.init(
    # Set entity to specify your username or team name
    # ex: entity="carey",
    # Set the project where this run will be logged
    project="685",
    entity="etower",
    config=config,
    mode="online" if config.use_wandb else "disabled"
)


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
log_file = join(cwd, "Logs/", f"{config.model_name}.log")

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

# create an instance of the experiment runner class
# num_examples, num_test, users_path, data_path, min_range, max_index
exp_runner = ExperimentRunner(config.num_examples, config.num_test, config.users_path, config.data_path, config.min_range, config.max_range, use_wandb=config.use_wandb)

# create the model(s) you are going to evaluate
#     data_path, shared_docs_path, num_examples
model = T5_Base(config.data_path, config.shared_docs_path, config.num_examples, config.finetune,
                config.data_aug, config.aug_path, config.gamma, use_wandb=config.use_wandb, verbose=config.verbose, num_aug=config.num_aug)

# perform the actual experiment
#   Could loop over several models/params. In Models, can 
#   define extra params for stuff as needed and then instantiate
#   as needed.                                                                   
#
#     model, num_trials, save_results (print results to log file), model_name, exp_folder=None, multi_processing=True // artifcact, huggingface API does not allow for multi-processing
exp_runner.get_model_analysis(model, config.num_trials, True, config.model_name, multi_processing=False, use_wandb=config.use_wandb)

if config.use_wandb:
    wandb.finish()
