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
