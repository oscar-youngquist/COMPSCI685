from Models.Model import Model
import numpy as np
import pandas as pd
import logging
from utils.filtering import Sentence_Prefilter_Wrapper
from utils.utils import get_sentences, get_avg_example_length, suppress_stdout, set_global_logging_level
from utils.model_training import fine_tune_model, fine_tune_model_aug
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments
import torch
# from torch.utils.data import DataLoader
import gc
import logging
set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])
import os
from os.path import join
import wandb


model_path = join(os.path.dirname(os.path.realpath(__file__)), *["saved_models", "t5-small"])

class T5_Base(Model):

    # constructor (obviously). Of course you can add anyother necessary nonsense as params
    #    and pass them in accordingly from the Exp/script.

    # NOTE: New parameter - finetune, a command-line arg for exp script to use basic (no data augmentation or wiki-pretraining) finetuning
    def __init__(self, data_path, shared_docs_path, num_examples, epochs=30, lr=5e-6, finetune=False, data_aug=False, aug_path="", gamma=1.0, use_wandb=False, verbose=False, num_aug=None):
        super().__init__(data_path, shared_docs_path, num_examples)
        self.verbose = verbose
        self.use_wandb = use_wandb
        self.df = pd.read_csv(self.data_path)
        self.data_aug = data_aug
        self.aug_path = aug_path
        self.num_aug = num_aug
        self.filter_obj = Sentence_Prefilter_Wrapper(data_path, shared_docs_path)

        # Download T5 every time instead of having to manually git clone
        #self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        #self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

        # Requires cloning model
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)

        self.gamma = gamma
        self.finetune = finetune
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.p_num = 1
        self.num_pred = 0
        
        # I don't think these will change
        self.input_token_len = 512
        self.batch_size=5

        # these could all be passed into the constructor and iterated over gird-search style from
        #     a loop in the EXP script

        ##### BE SURE TO (in the EXP script) set the parameters to create new model names/exp folders for each configuration IMPORTANT NOTE !!!!!!!!!!!!!!!!!!!!!!! #####
        ##### namely, pass in an exp_folder name like: "t5_ft_hyperparam_lr_batchsize" and whenever you make a new model with a new set of parameters make the model 
        ##### change based on those params. Then the folders for storing results will be automatically handled                                                      #####
        ##### ALSO: always make sure you are updating the name of the log file for any experiment                                                      #####
        self.lr = lr
        self.adam_ep = 1e-8
        self.finetune_epochs=epochs
        self.smoothing=0.05

        # Other hyperparameters (might not need to be changed)
        self.generation_num_beams = 1
        self.warmup_ratio = 0.1

        # should definitely look into all of these parameters
        # should also look into papers for param recommendations/tricks for fine-tuning Transfomers in general
        #     and T5 in particular 
        self.fine_tune_args = Seq2SeqTrainingArguments(
            output_dir="t5_trainer",
            do_train=True,
            learning_rate=self.lr,
            num_train_epochs=self.finetune_epochs,
            generation_num_beams=self.generation_num_beams,
            label_smoothing_factor=self.smoothing,
            per_device_train_batch_size=self.batch_size,
            warmup_ratio=self.warmup_ratio,
        )
        if self.use_wandb:
            wandb.config.update({"lr": self.lr,
                       "adam_epsilon": self.adam_ep,
                       "label_smoothing_factor": self.smoothing,
                       "num_train_epochs": self.finetune_epochs,
                       "generation_num_beams": self.generation_num_beams,
                       "batch_size": self.batch_size,
                       "warmup_ratio": self.warmup_ratio})


    # reset model to default state after. Be explicitly very clear about the data management. 
    def reset_model(self):
        self.model = self.model.to('cpu')
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        #self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

    # override abstract method
    def get_predicted_summary(self, target_doc, example_summaries, processed_ctr):
        # get the average length of the example in terms of 1) sentences and 2) words (tokens via nltk word_tokenize)
        avg_len, _ = get_avg_example_length(example_summaries, self.df)

        if self.finetune:
            if (self.num_pred % 3) == 0: # reset and train
                self.reset_model()

                if self.data_aug:
                    fine_tune_model_aug(trainer_args=self.fine_tune_args, model=self.model, tokenizer=self.tokenizer, token_len=self.input_token_len, lr=self.lr, adam_ep=self.adam_ep,
                                    batch_size=self.batch_size, epochs=self.finetune_epochs, example_summaries=example_summaries,
                                    sentence_prefilter=self.filter_obj.nearest_neighbor_bert_summary_filtering,
                                    prefilter_len=int(2*avg_len), df=self.df, aug_path=self.aug_path, gamma=self.gamma, device=self.device, use_wandb=self.use_wandb, num_aug=self.num_aug)
                else:
                    # function in utils/model_training.py that actual does the training of the
                    fine_tune_model(trainer_args=self.fine_tune_args, model=self.model, tokenizer=self.tokenizer, token_len=self.input_token_len, lr=self.lr, adam_ep=self.adam_ep,
                                    batch_size=self.batch_size, epochs=self.finetune_epochs, example_summaries=example_summaries,
                                    sentence_prefilter=self.filter_obj.nearest_neighbor_bert_summary_filtering,
                                    prefilter_len=int(2*avg_len), df=self.df, device=self.device)
                self.num_pred += 1

        summaries = [" ".join(summary['sentences']) for summary in example_summaries]

        output = self.tokenizer(summaries)
        avg_len = 0
        for token_arr in output['input_ids']:
            avg_len += len(token_arr)
        
        avg_token_len = avg_len/len(example_summaries)

        del summaries
        del output

        
        
        # use SBERT to get the most similar sentences to the target document: 2* the average length of the example summaries
        #     this is done as an initial pruning step and is something I have seen done a few times for long-passage abstractive summarization.
        #     If we are worried we can dig up some citations to justify if we want. 
        filtered_sentence_ids = self.filter_obj.nearest_neighbor_bert_summary_filtering(example_summaries=example_summaries, test_doc=target_doc, top_k=int(avg_len))

        # get the actual (in order) sentences from the target document
        target_doc_sentences = get_sentences(filtered_sentence_ids, target_doc, self.df)

        self.model.to(self.device)

        # this is all (literally) boiler plate copied and pasted from hugging face. Hopefully everything huggingface should be ~relatively~ this easy. 
        inputs = self.tokenizer([target_doc_sentences], max_length=self.input_token_len, truncation=True, return_tensors='pt').to(self.device)
        summary_ids = self.model.generate(inputs['input_ids'], max_length=int(avg_token_len), early_stopping=True)
        sentences = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]     

        inputs.to('cpu')
        torch.cuda.empty_cache()  

        prediction = " ".join(sentences)
        if self.verbose:
            print("avg token len: ", avg_token_len)
            print(prediction)
            print(f"FINISHED PREDICTION {self.p_num}")
        self.p_num += 1
        return prediction
