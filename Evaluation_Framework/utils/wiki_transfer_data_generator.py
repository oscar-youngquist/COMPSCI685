import torch
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import json
import random
from os.path import join
from utils import get_avg_example_length


class WikiTransferDataGenerator():

    def __init__(self, shared_docs_path):
        self.shared_docs_path = shared_docs_path
        with open(join(self.shared_docs_path, "state_indicies.txt")) as f:
            file = f.read()
            f.close()
        self.doc_indices = json.loads(file)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # hard-coding (gross) for now, might change later
        self.file_ranges = [i for i in range(0, 13)]
        self.outputfile = "../../Wiki_Dump/ProcessedArticles/split_jsons/jsons_{}.txt"


    def generate_synthetic_articles(self, example_summaries, set_size, window_size):
        training_set = [[], []]

        # get the average length of these examples
        avg_example_len, _ = get_avg_example_length(example_summaries, None)
        
        # list to hold the comparsion data from the example summaries
        example_embeddings = []

        # iterate over all the example summaries
        for ex_sum in example_summaries:
            # extract all the relevant info for this example summary
            sentence_ids = [int(s) for s in ex_sum["sentence_ids"]]
            doc_name = ex_sum['state_name']
            
            # if the similarity measure is SBERT cosine similarity
            npz_path = join(self.shared_docs_path, "ProcessedStateDocuments/", doc_name.strip() + "sudocu.npz")
            data = np.load(npz_path)
            sentence_embeddings = data['embedding']
                
            ex_doc_idx_offset = self.doc_indices[doc_name][0]

            example_embedding = []
                
            # iterate over every sentence id in this example, and extract the 
            #     appropriate SBERT sentence embeddings
            for s_id in sentence_ids:
                adjusted_index = s_id - ex_doc_idx_offset
                example_embedding.append(sentence_embeddings[adjusted_index])

            example_embeddings.append(np.mean(np.array(example_embedding), axis=0))
        
        # end example summary for-loop

        # get the average SBERT embedding of the example summaries
        avg_example_embedding = np.mean(np.array(example_embeddings), axis=0)

        # get the threshold (building now, might only use later however)
        threshold_sim = 0
        for ex in example_embeddings:
            # 1 - cosine distance = cosine similarity 
            threshold_sim += 1 - cosine(ex, avg_example_embedding)

        threshold_sim = threshold_sim / len(example_embeddings)

        print(threshold_sim)

        # document counter
        doc_ = 0
        while (len(training_set[0]) < set_size and doc_ < 14):
            
            # print("opeing: " + self.outputfile.format(doc_))
            with open(self.outputfile.format(doc_)) as f:
                for json_obj in f:
                    if (len(training_set[0]) > set_size): break
                    # if (len(training_set[0]) % 1000 == 0): print("completed: ", len(training_set[0]))
                    article_json = json.loads(json_obj)
                    article_length = article_json['length']
                    article_sentences = article_json['sentences']
                                        
                    # get embeddings 
                    sbert_article_embeddings = self.model.encode(sentences=article_sentences, device=self.device, batch_size=128, convert_to_numpy=True).tolist()

                    similarity_scores = []
                    for embedding in sbert_article_embeddings:
                        similarity_scores.append(1 - cosine(embedding, avg_example_embedding))

                    # sort the indices
                    indices = np.argsort(similarity_scores)
                    # we want descending order
                    indices = indices[::-1]

                    # get the top most similar sentences for the synthetic summary
                    summary_ids = indices[:int(avg_example_len)]

                    # get the context ids
                    context_ids = []

                    # now randomly select two sentences to include in the "context" from the set
                    #    [top_k_id - window_size ... top_k_id ... top_k_id + window_size]. 
                    for id in summary_ids:
                        min_range = 0 if (id - window_size < 0) else id - window_size
                        max_range = article_length-1 if (id + window_size > article_length-1) else id + window_size
                        context_ids.extend(random.sample(range(min_range, max_range), 2))

                    article_sentences = np.array(article_sentences)
                    
                    # now build the actual summary and article
                    summ =  " ".join(article_sentences[summary_ids])
                    context = " ".join(article_sentences[context_ids])

                    # print(summ)
                    # print(context)
                    avg_syn_summ = np.mean(np.array(sbert_article_embeddings)[summary_ids], axis=0)
                    simm  = 1 - cosine(avg_syn_summ, avg_example_embedding)

                    if (simm > 0.3):
                        # print(simm)
                        training_set[0].append(summ)
                        training_set[1].append(context)
                doc_ += 1

                f.close
        print(len(training_set[0]))
        
        return training_set