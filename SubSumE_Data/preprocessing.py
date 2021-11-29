import pandas as pd
import json
import numpy as np
import nltk
import os
import torch
from sentence_transformers import SentenceTransformer
import re


class DB:
    """
    Populates @data: a database (list of dictionaries) with schema (pid, name, sid, sentence, score) where sid is the primary key
    """

    def __init__(
        self,
        folder_path,
        base_path,
        shared_docs_path,
        abbreviation_replacer,
    ):
        
        # path to the collected data
        self.folder_path = folder_path
        
        # base_path for all the output produced during preprocessing
        self.base_path = base_path
        
        # path for state_documents folder within the preprocessing_output folder
        self.state_docs_path = base_path + "StateDocuments/"
        
        # path to the folder that contains the shared documents the preprocessing for all
        #     topic model experiments will need
        self.shared_docs_path = shared_docs_path
        
        # SBERT model used to encode the sentences from each document
        #     side note: the specific SBERT model being loaded below
        #                is the one recommended by the creators of the 
        #                SentenceTransformer's package for cosine-similairty calculations
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # if the folder for the state txt documents does not exist create it
        if not os.path.exists(self.state_docs_path):
            os.mkdir(self.state_docs_path)
            
        self.abbreviation_replacer = abbreviation_replacer
        self.data = []
        
        # extract lowercase strings of statenames for use while processing 
        self.state_names = [
            line.strip().lower()
            for line in open(self.shared_docs_path + "just_states.txt").readlines()
            if line.strip() != ""
        ]
        
        # actually performt the processing of the sate documents
        self.process()

    def process(self):
        
        # create two lists of stopwwords, one from NLTK and one that was predefined 
        stopwords = nltk.corpus.stopwords.words("english")
        stopwords2 = [
            line.strip().lower()
            for line in open(self.shared_docs_path + "stopwords.txt").readlines()
            if line.strip() != ""
        ]
        
        # counters use to keep track of state number (pid) and 
        #     the "gloabl" sid number. Gloabl in reference to the 
        #     indexing in the final CSV of all processed sentences
        pid, sid_global = 0, 0
        
        # dictonary used to track the starting index for each state's collection of
        #     sentence in the complete processed CSV
        state_meta_data = {}
        
        # iterate over every state
        for state_name in self.state_names:
            # read in the state json files from the Site repo
            file_path = self.folder_path + state_name + ".json"
            f = open(file_path,)
            data = json.load(f)
            f.close()
            
            # variable to hold all the state sentences
            #     then split into sentences according the NLTK's
            #     sent_tokenize method
            #     These sentences are used by non-SuDocu methods such as 
            #     TestRank, Keyword, etc...
            state_doc = ""
            state_doc_embedding = []
            
            # BERT embeddings for sudocu. Currently used by the
            #     nearest neighbor BERT based generator
            state_doc_embedding_sudocu = []
            
        
            # loop variable used for storing the state index metadata
            first_loop = True
            
            # iterate over every sentence in the state document
            for sentence in data:
                # extract the actual text from the current sentence
                text = sentence['sentenceText']
                
                # add the current sentence's text to the total
                #     state document variable
#                 state_doc += text + " "
                
    
                # if this is the first loop then save the relavent state metadata
                if first_loop:
                    first_loop = False
                    # this sid is in reference to the document the sentence 
                    #     is coming from
                    sid = sentence['sentenceNumber']
                    state_meta_data[state_name] = (sid_global, sid)
                
                
                # if any abbreviations are present, replace them
                for ab in self.abbreviation_replacer:
                    text = re.sub(ab[0], ab[1], text)
                
                # add the BERT embedding 
                state_doc_embedding_sudocu.append(self.model.encode(text.lower(), device=self.device, batch_size=128, convert_to_numpy=True).tolist())
                
                
                # row = {"pid": pid, "name": state_name, "sid": sid_global, "sentence": text}
                # self.data.append(row)
                sid_global += 1
            pid += 1

#             with open(self.state_docs_path + state_name + ".txt", 'w') as outfile:
#                 outfile.write(state_doc)
#                 outfile.close()
            
            
            # tokenize the sentences from the currently being processed 
            #     state document. 
#             sen_array = sent_tokenize(state_doc)
            
            # for each tokenized sentence, append the BERT embedding for that 
            #     that sentence. This data is used by the BERT baseline. 
#             for sen in sen_array:
#                 state_doc_embedding.append(self.model.encode(sen.lower()))
            
            # save off each BERT embedded document. One for nearest neighbor generation and one for the
            #     BERT baseline. 
#             np.savez(self.state_docs_path + state_name, embedding=np.array(state_doc_embedding))
            np.savez(self.state_docs_path + state_name + "sudocu", embedding=np.array(state_doc_embedding_sudocu))
                
        # generate the merit score for each sentence
        # scores = self.get_sentence_score()

        # for each sentence add the merit score to the 
        # for i in range(len(self.data)):
        #     self.data[i]["score"] = round(scores[self.data[i]["sid"]], 2)
        
        # save the extracted state meta_data for use by the user_summary_extraction.py script
        # with open(self.base_path + 'state_indicies.txt', 'w') as outfile:
        #     json.dump(state_meta_data, outfile)
            

    def get_all_sentences(self):
        # append a tuple containing the sid and sentence text
        #    for each sentence in the procssed data
        sentences = []
        for row in self.data:
            sentences.append((row["sid"], row["sentence"]))
        return sentences

    def show_data(self):
        for row in self.data:
            print(row)

    def get_sentence_score(self):
        """
        Computes sentence scores using word frequencies. Sentences that contain frequently occur words (except stop words)
        are good candidate for including in the summary.
        :param sentences:
        :return:
        """
        # extract all sentences from the procssed sentence in self.data
        sentences = self.get_all_sentences()
        
        # clean each sentence
        cleaned_sentences = [
            (sentence_id, re.sub(r"\s+", " ", re.sub("[^a-zA-Z]", " ", sentence)))
            for (sentence_id, sentence) in sentences
        ]
        # join each cleaned sentence into a single string
        clean_text = "".join([sentence for (_, sentence) in cleaned_sentences])
        
        # set of english stopwords from NLTK
        stopwords = nltk.corpus.stopwords.words("english")
        
        word_frequencies = {}  # frequency of each word in the entire corpus
        
        # iterate over everyword in the corpus
        for word in nltk.word_tokenize(clean_text):
            # if the word is not a stopword, modify the frequency dict
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        # find the largest frequency value
        maximum_frequency = max(word_frequencies.values())

        # normalize each word's requency by the largest
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / maximum_frequency

        # dict to hold the merit score for each sentence
        sentence_scores = {}
        
        # iterate over every sid, text tuple in the cleaned sentences
        for (sid, sentence) in cleaned_sentences:
            sentence_scores[sid] = 0
            
            #iterate overy every word in the sentence
            for word in nltk.word_tokenize(sentence.lower()):
                
                # if this word is not a stop word, add it's freqency score to 
                #     sentences total score
                if word in word_frequencies.keys():
                    sentence_scores[sid] += word_frequencies[word]

        # extract the largest sentence score
        maximum_score = max(sentence_scores.values())
        
        # normalize each sentence's score by the largest sentence score
        for sid in sentence_scores.keys():
            sentence_scores[sid] = sentence_scores[sid] / maximum_score
        
        # return the calculate scores
        return sentence_scores


def main():
    abbreviation_replacer = [
        ["LL.B.", "LLB"],
        ["U.S.", "US"],
        ["Donald J. Trump", "Donald J Trump"],
        ["George H. W. Bush", "George H W Bush"],
        ["Sen.", "Senator"],
        ["H.R.", "HR"],
        ["\[[0-9]*\]", ""],
        [r"\s+", " "],
    ]
    
    nTopics = 10
    # will need to change once topic model experiments are done
    shared_docs_path = "/home/oscar/Research/SuDocu/Source/Shared_Documents/"
    base_path = os.getcwd() + "/preprocessing_output/"
    print(base_path)
    
    # check if the pre-processing output directory exists for this experiment
    #     if it does not, then create it
    if not os.path.exists(base_path):
            os.makedirs(base_path)
    
    db = DB(
        folder_path="/home/oscar/Research/SuDocu/Site/MTurk-data-collection/data scraping and processing/Data Scraping/jsons/",
        base_path= base_path,
        shared_docs_path=shared_docs_path,
        abbreviation_replacer=abbreviation_replacer,
    )
        
    # save total scores as NPZ indexed by model
    np.savez(base_path + "data",
             sudocu_counts=db.data)
    
    # tp = CTMTopicModeling(db.data, base_path, shared_docs_path, nTopics)
    # tp.store_topics_and_data()
    # tp.show_topics()
    # tp.show_data()


main()