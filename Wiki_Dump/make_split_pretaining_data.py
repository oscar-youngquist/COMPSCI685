# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util
from nltk import sent_tokenize
import random


# %%
model = SentenceTransformer('all-MiniLM-L6-v2')


base_path = "/home/oscar/Documents/CS685/AA/"
names = ["wiki_00", "wiki_01", "wiki_02", "wiki_03", "wiki_04",
        "wiki_05","wiki_06", "wiki_07", "wiki_08", "wiki_09",
        "wiki_10","wiki_11", "wiki_12", "wiki_13", "wiki_14"]

added_ids = []

for name in names:
   added_ids.append(set()) 


# %%
target_num = 100000
num_samples = 1000
save_size = 5000
added_num = 0
sample_id = 0
file_ctr = 0

outputfile = "ProcessedArticles/split_jsons/jsons_{}.txt"
outputfile_npz = "ProcessedArticles/split_jsons/embeddings_{}.npz"

embeddings = []

while (added_num < target_num):
    sample_ctr = 0
    doc_ = random.choice(names)
    
    temp_list = []

    with open(base_path + doc_) as f:
        for json_obj in f:
            temp_list.append(json_obj)
        f.close()

    jsons_file = open(outputfile.format(file_ctr), "w")

    print(doc_)

    while(sample_ctr < num_samples):
        sample = json.loads(random.choice(temp_list))

        sample_id = sample['id']
        text = sample['text']

        if sample_id in added_ids[names.index(doc_)]:
            continue

        if (len(text) < 10000):
            continue

        sample = {'id':sample_id}
        sample_sentences = []
        sample_sentence_embeddings = []

        text_sentences = sent_tokenize(text=text)

        sbert_embeddings = model.encode(sentences=text_sentences, batch_size=128, convert_to_numpy=True).tolist()


        for sent, sent_embed in zip(text_sentences, sbert_embeddings):
            if len(sent) > 2 and sent is not None:
                sample_sentences.append(sent)
                sample_sentence_embeddings.append(sent_embed)

        sample['length'] = len(sample_sentences)
        sample['sentences'] = sample_sentences

        embeddings.append(sample_sentence_embeddings)
        

        sample_json_encoded = json.dumps(sample)

        added_ids[names.index(doc_)].add(sample_id)

        jsons_file.write(sample_json_encoded + "\n")

        sample_ctr += 1
        added_num += 1

        if added_num % save_size == 0:
            np.savez_compressed(outputfile_npz.format(file_ctr), embeddings=np.array(embeddings))
            embeddings = []
            jsons_file.close()
            file_ctr += 1
            jsons_file = open(outputfile.format(file_ctr), "x")


    print("num completed: %d" % added_num)
    jsons_file.close()


# %%



