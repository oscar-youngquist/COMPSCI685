from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AdamW
import torch
from torch.utils.data import DataLoader, dataloader
import gc
from utils.utils import get_sentences, get_avg_example_length, set_global_logging_level
import torch.nn.functional as F

class SubSumEDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['decoder_input_ids'] = torch.tensor(self.labels['input_ids'][idx])
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def kl_for_log_probs(log_p, log_q):
    p = torch.exp(log_p)
    neg_ent = torch.sum(p * log_p, axis=-1)
    neg_cross_ent = torch.sum(p * log_q, axis=-1)
    kl = neg_ent - neg_cross_ent
    return kl

def tokenize_batch(tokenizer, token_len, sentences, output=False):
    if output:
        with tokenizer.as_target_tokenizer():
            return tokenizer(sentences, max_length=token_len, padding=True, truncation=True)
    return tokenizer(sentences, max_length=token_len, padding=True, truncation=True)

def build_datasets(tokenizer, token_len, sentence_prefilter, prefilter_len, example_summaries, df):
        state_names = [summary['state_name'] for summary in example_summaries]
        texts = []
        avg_len, _ = get_avg_example_length(example_summaries, df)
        
        for state_name in state_names: 
            filtered_sentence_ids = sentence_prefilter(example_summaries=example_summaries, test_doc=state_name, top_k=prefilter_len)
            trimmed_document = get_sentences(filtered_sentence_ids, state_name, df)
            texts.append(trimmed_document)
            # print(trimmed_document)

        summaries = [" ".join(summary['sentences']) for summary in example_summaries]

        texts = tokenize_batch(tokenizer, token_len, texts)
        summaries = tokenize_batch(tokenizer, token_len, summaries, output=True)

        # for token_arr in texts['input_ids']:
        #     print(np.array(token_arr).shape)
        # print(summaries)

        train_dataset = SubSumEDataset(texts, summaries)
        # train_dataset = {'input_ids': texts, 'labels': summaries}
        return train_dataset

# currently uses Huggingfaces seq2seq trainer to train model, but there is commented out code for training with a natice pytorch loop
# feel free to pick either. I picekd trainer because it must have built in method to stablize training that we could benefit from. 
def fine_tune_model(trainer_args, model, tokenizer, token_len, lr, adam_ep, batch_size, epochs, example_summaries, sentence_prefilter, prefilter_len, df, device):
    # lr = 5e-6
    # smoothing = 0.1
    # epochs = 30
    # beam_size = 1
    
    
    model.to(device)

    train_dataset = build_datasets(tokenizer, token_len, sentence_prefilter, prefilter_len, example_summaries, df)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 5

    # optim = AdamW(model.parameters(), lr=lr, eps=adam_ep) #1e-8
    
    t = Seq2SeqTrainer(model=model, args=trainer_args, train_dataset=train_dataset)

    t.train()

    # for epoch in range(epochs):
    #     for batch in train_loader:
    #         t.training_step(model=model, input=batch)

    #         # optim.zero_grad()
    #         # input_ids = batch['input_ids'].to(device)
    #         # attention_mask = batch['attention_mask'].to(device)
    #         # labels = batch['labels'].to(device)

    #         # outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    #         # loss = outputs[0]
    #         # loss.backward()
    #         # optim.step()
    #         # input_ids.to('cpu')
    #         # attention_mask.to('cpu')
    #         # labels.to('cpu')
    #         torch.cuda.empty_cache()

    model.to('cpu')
    torch.cuda.empty_cache()

    del train_dataset
    # del train_loader
    # del optim
    gc.collect()

# currently uses Huggingfaces seq2seq trainer to train model, but there is commented out code for training with a natice pytorch loop
# feel free to pick either. I picekd trainer because it must have built in method to stablize training that we could benefit from.
def fine_tune_model_aug(trainer_args, model, tokenizer, token_len, lr, adam_ep, batch_size, epochs, example_summaries,
                    sentence_prefilter, prefilter_len, df, df_aug, gamma, device):
    model.to(device)

    # Build datasets the same way but with different df of augmented sentences
    train_dataset = build_datasets(tokenizer, token_len, sentence_prefilter, prefilter_len, example_summaries, df)
    train_dataset_aug = build_datasets(tokenizer, token_len, sentence_prefilter, prefilter_len, example_summaries, df_aug)

    # Custom dataloader that loads 1 batch of input data as well as 1 batch of augmented data
    # To do larger augmented batch size (e.g., 10 augmented sentences), simply add train_dataset_aug1, aug2, ...
    # Would have to change training loop slightly in order to calculate loss on multiple batches of augmented data
    train_loader_aug = torch.utils.data.DataLoader(
             AugmentedDataset(
                 train_dataset,
                 train_dataset_aug
             ),
             batch_size=batch_size, shuffle=True)
    optim = AdamW(model.parameters(), lr=lr, eps=adam_ep) #1e-8

    for epoch in range(epochs):
        for (batch, batch_aug) in train_loader_aug:
            optim.zero_grad()

            # Input data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            # Regular T5 supervised loss
            with torch.no_grad():
                supervised_loss = outputs['loss']

            # TODO: in case of multiple augmented examples per input example, concatenate all of these to one larger batch
            # Augmented data
            input_ids = batch_aug['input_ids'].to(device)
            attention_mask = batch_aug['attention_mask'].to(device)
            labels = batch_aug['labels'].to(device)
            outputs_aug = model(input_ids, attention_mask=attention_mask, labels=labels)

            # Consistency loss: KL divergence between distribution on augmented and original input
            input_log_probs = F.log_softmax(outputs['logits'], dim=-1)
            aug_log_probs = F.log_softmax(outputs_aug['logits'], dim=-1)
            consistency_loss = kl_for_log_probs(input_log_probs, aug_log_probs)
            consistency_loss = torch.sum(consistency_loss) # UDA paper does mean, zero shot BART paper does sum

            loss = supervised_loss + gamma * consistency_loss
            loss.backward() # TODO: check that grad doesn't propagate through to supervised examples? No_grad should work though
            optim.step()

            input_ids.to('cpu')
            attention_mask.to('cpu')
            labels.to('cpu')
            torch.cuda.empty_cache()

    model.to('cpu')
    torch.cuda.empty_cache()

    del train_dataset
    del train_loader_aug
    del optim
    gc.collect()
