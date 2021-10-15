from Models.Model import Model
import numpy as np
import pandas as pd
from nltk import word_tokenize
import logging
from utils.filtering import Sentence_Prefilter_Wrapper
from transformers import PegasusTokenizer, PegasusForConditionalGeneration


class Pegasus_Base(Model):

    # constructor (obviously). Of course you can add anyother necessary nonsense as params
    #    and pass them in accordingly from the Exp/script.
    def __init__(self, data_path, shared_docs_path, num_examples):
        super().__init__(data_path, shared_docs_path, num_examples)
        self.df = pd.read_csv(self.data_path)
        self.filter_obj = Sentence_Prefilter_Wrapper(data_path, shared_docs_path)
        self.model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-cnn_dailymail').to('cuda')
        self.tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-cnn_dailymail')


    # override abstract method
    def get_predicted_summary(self, target_doc, example_summaires, processed_ctr):

        # get the average length of the example in terms of 1) sentences and 2) words (tokens via nltk word_tokenize)
        avg_len, avg_token_len = self.get_avg_example_length(example_summaires)

        # use SBERT to get the most similar sentences to the target document: 2* the average length of the example summaries
        #     this is done as an initial pruning step and is something I have seen done a few times for long-passage abstractive summarization.
        #     If we are worried we can dig up some citations to justify if we want. 
        filtered_sentence_ids = self.filter_obj.nearest_neighbor_bert_summary_filtering(example_summaries=example_summaires, test_doc=target_doc, top_k=int(2*avg_len))
        
        # get the actual (in order) sentences from the target document
        target_doc_sentences = self.get_sentences(filtered_sentence_ids, target_doc)
        
        # this is all (literally) boiler plate copied and pasted from hugging face. Hopefully everything huggingface should be ~relatively~ this easy. 
        inputs = self.tokenizer([target_doc_sentences], max_length=1024, truncation=True, return_tensors='pt').to('cuda')
        summary_ids = self.model.generate(inputs['input_ids'], max_length=int(avg_token_len), early_stopping=True)
        sentences = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]       
        
        prediction = " ".join(sentences)
        # print(prediction)
        return prediction


    def get_avg_example_length(self, example_summaries):
        avg_sentence_len = 0.0
        avg_word_len = 0.0

        # iterate over every example summary
        for i in range(len(example_summaries)):
            # get the example currently being processed
            ex = example_summaries[i]

            sentence_ids = [int(s) for s in ex["sentence_ids"]]
            
            # keep track of the lengths of the example summaries
            avg_sentence_len += float(len(sentence_ids))
            
            for s_id in sentence_ids:
                sentence = self.df[self.df['sid'] == s_id]['sentence'].to_numpy()[0]
                avg_word_len += len(word_tokenize(sentence))

        # get the min and max of every topic from across all the summaries
        avg_sentence_len = avg_sentence_len / len(example_summaries)
        avg_word_len = avg_word_len / len(example_summaries)
            
        return (avg_sentence_len, avg_word_len)
    
    def get_sentences(self, sentence_ids, target_doc):
        # we need to actually retrieve the literal text sentences
        return " ".join(self.df[(self.df['name'] == target_doc) & (self.df['sid'].isin(sentence_ids))]['sentence'].tolist())
