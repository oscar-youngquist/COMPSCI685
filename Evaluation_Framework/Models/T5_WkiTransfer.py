from Models.Model import Model
import numpy as np
import pandas as pd
import logging
from utils.filtering import Sentence_Prefilter_Wrapper
from utils.utils import get_sentences, get_avg_example_length, suppress_stdout, set_global_logging_level
from utils.model_training import wiki_trasnfer_training, fine_tune_model, fine_tune_model_aug
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
import torch
# from torch.utils.data import DataLoader
import gc
import logging
set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])
import os
from os.path import join


model_path = join(os.path.dirname(os.path.realpath(__file__)),"saved_models/t5-small")

class T5_Wiki(Model):

    # constructor (obviously). Of course you can add anyother necessary nonsense as params
    #    and pass them in accordingly from the Exp/script.

    # NOTE: New parameter - finetune, a command-line arg for exp script to use basic (no data augmentation or wiki-pretraining) finetuning
    def __init__(self, data_path, shared_docs_path, num_examples, wiki_path, lr, bs, epchs):
        super().__init__(data_path, shared_docs_path, num_examples)
        self.df = pd.read_csv(self.data_path)
        self.filter_obj = Sentence_Prefilter_Wrapper(data_path, shared_docs_path)
        
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.p_num = 1
        self.wiki_path = wiki_path
        
        # I don't think these will change
        self.input_token_len = 512
        self.num_pred = 0

        # these could all be passed into the constructor and iterated over gird-search style from
        #     a loop in the EXP script

        ##### BE SURE TO (in the EXP script) set the parameters to create new model names/exp folders for each configuration IMPORTANT NOTE !!!!!!!!!!!!!!!!!!!!!!! #####
        ##### namely, pass in an exp_folder name like: "t5_ft_hyperparam_lr_batchsize" and whenever you make a new model with a new set of parameters make the model 
        ##### change based on those params. Then the folders for storing results will be automatically handled                                                      #####
        ##### ALSO: always make sure you are updating the name of the log file for any experiment                                                      #####
        self.lr =lr
        self.adam_ep = 1e-8
        self.finetune_epochs=epchs
        self.smoothing=0.1
        self.batch_size=bs


        # should definitely look into all of these parameters
        # should also look into papers for param recommendations/tricks for fine-tuning Transfomers in general
        #     and T5 in particular 
        self.wiki_args = Seq2SeqTrainingArguments(
            output_dir="t5_trainer_wiki",
            do_train=True,
            learning_rate=self.lr,
            num_train_epochs=self.finetune_epochs,
            generation_num_beams=1,
            label_smoothing_factor=self.smoothing,
            per_device_train_batch_size=self.batch_size,
            warmup_ratio=0.1, 
            # lr_scheduler_type='polynomial', use if lr < 1e-7
            fp16=True,
            evaluation_strategy="epoch",
            save_strategy="no"
        )


        self.fine_tune_args = Seq2SeqTrainingArguments(
            output_dir="t5_trainer",
            do_train=True,
            learning_rate=5e-6,
            num_train_epochs=20,
            generation_num_beams=1,
            label_smoothing_factor=self.smoothing,
            per_device_train_batch_size=2,
            warmup_ratio=0.1,
            save_strategy="no" 
        )


    # reset model to default state after. Be explicitly very clear about the data management. 
    def reset_model(self):
        self.model = self.model.to('cpu')
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

    # override abstract method
    def get_predicted_summary(self, target_doc, example_summaries, processed_ctr):
        # get the average length of the example in terms of 1) sentences and 2) words (tokens via nltk word_tokenize)
        avg_len, _ = get_avg_example_length(example_summaries, self.df)

        if (self.num_pred == 0):
          self.reset_model()
          # function in utils/model_training.py that actual does the training of the
          wiki_trasnfer_training(self.wiki_args, self.model, processed_ctr, self.wiki_path,
              self.tokenizer, self.input_token_len, self.device)

        #   fine_tune_model(trainer_args=self.fine_tune_args, model=self.model, tokenizer=self.tokenizer, token_len=self.input_token_len, lr=self.lr, adam_ep=self.adam_ep,
        #                     batch_size=self.batch_size, epochs=self.finetune_epochs, example_summaries=example_summaries,
        #                     sentence_prefilter=self.filter_obj.nearest_neighbor_bert_summary_filtering,
        #                     prefilter_len=int(2*avg_len), df=self.df, device=self.device)

          # fine_tune_model_aug(trainer_args=self.fine_tune_args, model=self.model, tokenizer=self.tokenizer, token_len=self.input_token_len, lr=self.lr, adam_ep=self.adam_ep,
          #                       batch_size=self.batch_size, epochs=self.finetune_epochs, example_summaries=example_summaries,
          #                       sentence_prefilter=self.filter_obj.nearest_neighbor_bert_summary_filtering,
          #                       prefilter_len=int(2*avg_len), df=self.df, aug_path=self.aug_path, gamma=self.gamma, device=self.device, use_wandb=self.use_wandb, num_aug=self.num_aug)
                            
          self.num_pred += 1
        else:
          self.num_pred += 1
        
        if self.num_pred == 3:
          self.num_pred = 0

        summaries = [" ".join(summary['sentences']) for summary in example_summaries]

        output = self.tokenizer(summaries)
        avg_len = 0
        for token_arr in output['input_ids']:
            avg_len += len(token_arr)
        
        avg_token_len = avg_len/len(example_summaries)

        del summaries
        del output        

        self.model.to(self.device)
        
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
        self.model.to('cpu')
        torch.cuda.empty_cache()  

        prediction = " ".join(sentences)
        print(prediction)
        print(f"FINISHED PREDICTION {self.p_num}")
        self.p_num += 1
        return prediction