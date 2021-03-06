from Models.Model import Model
import numpy as np
import pandas as pd
import logging
from utils.filtering import Sentence_Prefilter_Wrapper
from utils.utils import get_sentences, get_avg_example_length, set_global_logging_level
from utils.model_training import fine_tune_model
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, T5ForConditionalGeneration, Seq2SeqTrainingArguments
import torch
# from torch.utils.data import DataLoader
import gc
import logging
set_global_logging_level(logging.ERROR, ["transformers", "nlp", "torch", "tensorflow", "tensorboard", "wandb"])


model_path = "google/pegasus-xsum"



class Pegasus_Base(Model):

    # constructor (obviously). Of course you can add anyother necessary nonsense as params
    #    and pass them in accordingly from the Exp/script.
    def __init__(self, data_path, shared_docs_path, num_examples, finetune=False):
        super().__init__(data_path, shared_docs_path, num_examples)
        self.df = pd.read_csv(self.data_path)
        self.filter_obj = Sentence_Prefilter_Wrapper(data_path, shared_docs_path)
        
        self.model = PegasusForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = PegasusTokenizer.from_pretrained(model_path)
        
        self.finetune = finetune
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.p_num = 1
        self.input_token_len = 256
        self.lr = 5e-6
        self.adam_ep = 1e-8
        self.batch_size=5
        self.finetune_epochs=30
        self.smoothing=0.05


        # should definitely look into all of these parameters
        # should also look into papers for param recommendations/tricks for fine-tuning Transfomers in general
        #     and Pegasus in particular 
        self.fine_tune_args = Seq2SeqTrainingArguments(
            output_dir="pegasus_trainer",
            do_train=True,
            learning_rate=self.lr,
            num_train_epochs=self.finetune_epochs,
            generation_num_beams=1,
            label_smoothing_factor=self.smoothing,
            per_device_train_batch_size=self.batch_size,
            warmup_ratio=0.1, 
        )


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

        if self.finetune:
            self.reset_model()
            fine_tune_model(trainer_args=self.fine_tune_args, model=self.model, tokenizer=self.tokenizer, token_len=self.input_token_len, lr=self.lr, adam_ep=self.adam_ep,
                            batch_size=self.batch_size, epochs=self.finetune_epochs, example_summaries=example_summaries,
                            sentence_prefilter=self.filter_obj.nearest_neighbor_bert_summary_filtering,
                            prefilter_len=int(2*avg_len), df=self.df, device=self.device)

        summaries = [" ".join(summary['sentences']) for summary in example_summaries]

        output = self.tokenizer(summaries)
        avg_len = 0
        for token_arr in output['input_ids']:
            avg_len += len(token_arr)
        
        avg_token_len = avg_len/len(example_summaries)

        del summaries
        del output

        print(avg_token_len)
        
        
        # use SBERT to get the most similar sentences to the target document: 2* the average length of the example summaries
        #     this is done as an initial pruning step and is something I have seen done a few times for long-passage abstractive summarization.
        #     If we are worried we can dig up some citations to justify if we want. 
        filtered_sentence_ids = self.filter_obj.nearest_neighbor_bert_summary_filtering(example_summaries=example_summaries, test_doc=target_doc, top_k=int(avg_len))

        # get the actual (in order) sentences from the target document
        target_doc_sentences = get_sentences(filtered_sentence_ids, target_doc, self.df)

        self.model.to(self.device)

        # this is all (literally) boiler plate copied and pasted from hugging face. Hopefully everything huggingface should be ~relatively~ this easy. 
        inputs = self.tokenizer([target_doc_sentences], max_length=126, truncation=True, return_tensors='pt').to(self.device)
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
