from Models.Model import Model
import numpy as np
import pandas as pd


class RandomN(Model):

    # constructor (obviously). Of course you can add anyother necessary nonsense as params
    #    and pass them in accordingly from the Exp/script.
    def __init__(self, data_path, shared_docs_path, num_examples):
        super().__init__(data_path, shared_docs_path, num_examples)
        self.df = pd.read_csv(self.data_path)

    # override abstract method
    def get_predicted_summary(self, target_doc, example_summaries, processed_ctr):
        avg_len = self.get_avg_example_length(example_summaries)
        N = int(np.ceil(avg_len))
        return self.get_randomN(target_doc, N)

    def get_avg_example_length(self, example_summaries):
      avg_sentence_len = 0.0

      # iterate over every example summary
      for i in range(len(example_summaries)):
          # get the example currently being processed
          ex = example_summaries[i]
          sentence_ids = [int(s) for s in ex["sentence_ids"]]
            
          # keep track of the lengths of the example summaries
          avg_sentence_len += float(len(sentence_ids))

      # get the min and max of every topic from across all the summaries
      avg_sentence_len = avg_sentence_len / len(example_summaries)
      return avg_sentence_len

    def get_randomN(self, target_doc, N):
      state_df = self.df[self.df['name'] == target_doc]
      state_df = state_df.sample(n=N)
      return " ".join(state_df['sentence'])

