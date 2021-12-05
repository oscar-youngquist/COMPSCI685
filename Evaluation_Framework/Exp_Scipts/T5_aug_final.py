import os
from os.path import join, exists
import sys
import logging
from pathlib import Path
from argparse import ArgumentParser, Namespace
import wandb


# test call: .py --finetune --data_aug --lr x --gamma x --epochs x

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
parser.add_argument('--wandb_username', type=str, default="")
args = parser.parse_args()

config = Namespace()
config.use_wandb = args.use_wandb
config.wandb_username = args.wandb_username
config.verbose = False
config.finetune = True
config.data_aug = True
config.num_aug = 10
config.gamma = 1.0
config.lr = 1e-6
config.epochs = 100

# experiment configuration
config.num_examples = 5
config.num_test = 3
config.shared_docs_path = os.path.join(Path(__file__).parent.parent.parent, "SubSumE_Data") # path to shared documents in dataset
config.aug_path = os.path.join(Path(__file__).parent.parent.parent, *["backtranslation", "paraphrases"]) # path to augmented sentences (backtranslation)
config.data_path = os.path.join(config.shared_docs_path, "processed_state_sentences.csv")      # path to the processed sentences csv
config.users_path = os.path.join(config.shared_docs_path, "Test")     # path to the misc. shared data (might not be needed anymore)
config.min_range = 0    # these min/max values are left-over from multi-processing experiments in which we would create n SuDocu models and then have each process (# of total user-summary instances)/n users. Thus we needed a min/max for the files to be read into memory by each model
config.max_range = 137
config.num_trials = 1    # number of times to evaluate all the examples
config.exp_folder = ""      # results folder for this experimental run, only used if running a) more than one model or b) the same model more than once
                                # model_name folder is added as a sub-folder to this one

# add your model names here
config.model_name = "T5_finetuned_aug_test"

if args.wandb_silent:
    os.environ["WANDB_SILENT"] = "true"

wandb.init(
    # Set entity to specify your username or team name
    # ex: entity="carey",
    # Set the project where this run will be logged
    project="685",
    entity=config.wandb_username,
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
log_file = join(cwd, "Logs/", f"{config.model_name}_lr_{config.lr}_epochs_{config.epochs}_test.log")

# set up logger
root = logging.getLogger()
root.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
root.addHandler(console_handler)

file_handler = logging.FileHandler(log_file, mode="w", encoding=None, delay=False)
file_handler.setLevel(logging.DEBUG)
root.addHandler(file_handler)

shared_docs_path = os.path.join(Path(__file__).parent.parent.parent, "SubSumE_Data")

# add the exception logging behavior to the logger
# sys.excepthook = exceptionhook

# create an instance of the experiment runner class
# num_examples, num_test, users_path, data_path, min_range, max_index
exp_runner = ExperimentRunner(config.num_examples, config.num_test, config.users_path, config.data_path, config.min_range, config.max_range, shared_docs_path, use_wandb=config.use_wandb)

# create the model(s) you are going to evaluate
#     data_path, shared_docs_path, num_examples
model = T5_Base(
    data_path=config.data_path,
    shared_docs_path=config.shared_docs_path,
    num_examples=config.num_examples,
    finetune=config.finetune,
    data_aug=config.data_aug,
    aug_path=config.aug_path,
    gamma=config.gamma,
    lr=config.lr,
    epochs=config.epochs,
    use_wandb=config.use_wandb,
    verbose=config.verbose,
    num_aug=config.num_aug
)

# perform the actual experiment
#   Could loop over several models/params. In Models, can 
#   define extra params for stuff as needed and then instantiate
#   as needed.                                                                   
#
#     model, num_trials, save_results (print results to log file), model_name, exp_folder=None, multi_processing=True // artifcact, huggingface API does not allow for multi-processing
exp_runner.get_model_analysis_final(model, True, config.model_name, multi_processing=False, use_wandb=config.use_wandb)

if config.use_wandb:
    wandb.finish()
