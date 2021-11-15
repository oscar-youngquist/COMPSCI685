import pandas as pd
from nltk import word_tokenize
import logging
import re
from contextlib import contextmanager
import sys, os

def get_avg_example_length(example_summaries, df):
    avg_sentence_len = 0.0
    avg_word_len = 0.0

    # iterate over every example summary
    for i in range(len(example_summaries)):
        # get the example currently being processed
        ex = example_summaries[i]

        sentences = ex["sentences"]

        # keep track of the lengths of the example summaries
        avg_sentence_len += float(len(sentences))

        for sentence in sentences:
            # sentence = df[df['sid'] == s_id]['sentence'].to_numpy()[0]
            avg_word_len += len(word_tokenize(sentence))

    # get the min and max of every topic from across all the summaries
    avg_sentence_len = avg_sentence_len / len(example_summaries)
    avg_word_len = avg_word_len / len(example_summaries)

    return (avg_sentence_len, avg_word_len)
    
def get_sentences(sentence_ids, target_doc, df):
    # we need to actually retrieve the literal text sentences
    return " ".join(df[(df['name'] == target_doc) & (df['sid'].isin(sentence_ids))]['sentence'].tolist())

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

# To control logging level for various modules used in the application:
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
