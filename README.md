# COMPSCI685

The Evaluation Framework folder contains code (distributed across a few sub-directories) for evaluating models on the SubSumE dataset.

The SubSumE folder contains the SubSumE dataset itself (both in the RAW and processed)

To create a new environment for this project: 
`conda create --name 685 python=3.6`

Install requirements using pip: `pip install transformers torch nltk spacy SentencePiece scipy scikit-learn rouge_score numpy pandas matplotlib sentence-transformers wandb`

Then, run the following commands:

`import nltk`

`nltk.download('punkt')`

` git lfs install`

`git clone https://huggingface.co/t5-small && mv t5-small ./Evaluation_Framework/Models/saved_models/t5-small`

To reproduce our results, do the click on the link to the Colab notebook below and do the following. 

LeadN: In the Evaluation_Framework/Exp_Scripts/ directory run the LeadN_EXP.py file

RandomN: In the Evaluation_Framework/Exp_Scripts/ directory run the RandomN_EXP.py file

T5_Baseline: In the Evaluation_Framework/Exp_Scripts/ directory run the file T5_Model_Exp.py file with the command line argument --finetune False

T5_Basic_Finetuning: In the Evaluation_Framework/Exp_Scripts/ directory run the file T5_Model_FineTuning_Test_Exp.py file

T5_Augmented_Finetuning: See the current command in the Colab notebook

T5_WikiTransfer: In the Evaluation_Framework/Exp_Scripts/ directory run the file T5_Wikitransfer_final.py file

T5_WikiTransfer with Basic Finetuning: In the Evaluation_Framework/Exp_Scripts/ directory run the file T5_Wikitransfer_final.py file with the command line argument --finetune_basic True

T5_WikiTransfer with Augmented Finetuning: In the Evaluation_Framework/Exp_Scripts/ directory run the file T5_Wikitransfer_final.py file with the command line argument --finetune_augmented True

Additionally, due to the size of the datasets created for the Wiki-Transfer datasets, we cannot upload them to github. Accordingly, they are n a Google drive that anyone with a UMass Amherst email can access. There are two splits for this data.

[Split one](https://drive.google.com/file/d/1_pb2EIeNg9Pgq-YSgQRRIg0CMyi184HR/view?usp=sharing)
[Split two](https://drive.google.com/file/d/1qIOddkv3cb6ZrnAXccoKrlHIz06HhGdX/view?usp=sharing)

To run the Wiki-Transfer experiments you need to 1) download BOTH of these splits 2) Create a Wiki_Dump/ProcessedArticles/test_example_datasets/ directory and place the first split in it and 3) Create a Wiki_Dump/ProcessedArticles/test_example_datasets_2/ directory and place the second split in it

## Demo training Notebook: ##

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VtpZhdWJff8ojAtSXG_gn8iL-ajUzIBW?usp=sharing)

======
