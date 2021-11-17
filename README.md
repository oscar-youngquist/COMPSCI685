# COMPSCI685

The Evaluation Framework folder contains code (distributed across a few sub-directories) for evaluating models on the SubSumE dataset.

The SubSumE folder contains the SubSumE dataset itself (both in the RAW and processed)

To create a new environment for this project: 
`conda create --name 685 python=3.6

Install requirements using pip: `pip install transformers torch nltk spacy SentencePiece scipy scikit-learn rouge_score numpy pandas matplotlib`

Then, run the following commands:

`import nltk`

`nltk.download('punkt')`

` git lfs install`

`git clone https://huggingface.co/t5-small && mv t5-small ./Evaluation_Framework/Models/saved_models/t5-small`

## Demo training Notebook: ##

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dSCFiSjTMFxotjPUdK4RSqtcSzbpg-fP?usp=sharing)


Wiki-Transfer update. 

Wiki_Dump folder contains a sub-folder, ProcessedArticles, where the wiki-transfer related data needs to be placed. This is very large however, so cannot go to github. Instead, I am placing a zipped version of the contexts of ProcessedArticles in the shared project google drive. (will add link when it finishes uploading). I modified the .gitignore file so that the ProcessedArticles folder is not tracked.

I added a function in utils/model_training.py that handles training either the first 20 training split examples (for dev) or all of the test set examples depening on the user_file_path passed in the exp script. Note, that either case requires the contents of ProcessedArticles. 
======
