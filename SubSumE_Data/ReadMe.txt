This folder contains the SubSumE dataset, both in it's RAW and processed forms. 

Raw_Wiki_Text_Files contains the raw txt files used to create this dataset. 

ProcessedStateDocuments contains the same txt file as the above folder, but also NPZ files which contain the SBERT embedding of every sentence in the underlying state document

user_summary_jsons contains actual summaries created by users in MTURK. Look in any of the json files to see how they are constructed and it will make sense. 

The Test and Train folders contain a "dev" and "test" split of the 275 individual (intent, 8 state documents, 8 example summaries) user entires. The split was done in the following manner:

Dataset Split: First, I split the dataset into a training and testing set. Each set contains 5 unique intents such that each set has two mostly objective/subjective intents and one balanced intent.
The training set contains 138 unique user interactions (and so 138*8=1104 example summaries) and the test set contains 137 interactions. The tables below break each set down into the number of
summaries per intent for each set.


Intent in Training Set
Number of Summaries
How is the weather of the state?
240
What is the state's policy regarding education?
232
What are the major historical events in this state?
224
Which places seem interesting to you for visiting in this state?
200
What are some of the most interesting things about this state?
208



Intent in Test Set
Number of Summaries
What drives the economy in this state?
232
What are the available modes of transport in this state?
240
How is the government structured in this state?
216
What about this state's arts and culture attracts you the most?
192
The main reasons why you would like living in this state
216

processed_state_sentences.csv contains every sentence in the entire corpus. Each sentence is accompanied with a "state-id" (pid), "state name" (name), "sentence-id" (sid), and the actual sentence text (sentence).


the state_indicies.txt (sic, I suck at spelling) contains the index of the first sentence for each state in the processed_state_sentences.csv file.   

state_names.txt contains the names and abbreviations for each state. I honestly do not remember if any of the evaluation framework code relies on this file, so I just kept it in. 




