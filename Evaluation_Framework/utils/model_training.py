from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AdamW
import torch
from torch.utils.data import DataLoader, dataloader
import gc
from utils.utils import get_sentences, get_avg_example_length, set_global_logging_level


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

#TODO: change code to use t5 df_aug as well as df
def build_datasets_aug(tokenizer, token_len, sentence_prefilter, prefilter_len, example_summaries, df):
    state_names = [summary['state_name'] for summary in example_summaries]
    texts = []
    avg_len, _ = get_avg_example_length(example_summaries, df)

    for state_name in state_names:
        filtered_sentence_ids = sentence_prefilter(example_summaries=example_summaries, test_doc=state_name,
                                                   top_k=prefilter_len)
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
                        sentence_prefilter, prefilter_len, df, device):
        model.to(device)

        train_dataset = build_datasets_aug(tokenizer, token_len, sentence_prefilter, prefilter_len, example_summaries, df)

        #TODO: do we want to do an entirely different train_dataset object?
        # We need to differentiate between augmented and regular examples, as grad flows through one but not the other

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 5

        optim = AdamW(model.parameters(), lr=lr, eps=adam_ep) #1e-8

        for epoch in range(epochs):
            for batch in train_loader: # TODO: change to "for batch, augmented batch"?
                optim.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs, logits = model(input_ids, attention_mask=attention_mask, labels=labels)

                # Regular T5 loss, we will calculate a different loss
                # loss = outputs
                # loss.backward()

                #TODO: implement consistency loss, add torch no grad for one of the components of the loss (check paper)

                optim.step()

                input_ids.to('cpu')
                attention_mask.to('cpu')
                labels.to('cpu')
                torch.cuda.empty_cache()

        model.to('cpu')
        torch.cuda.empty_cache()

        del train_dataset
        del train_loader
        del optim
        gc.collect()
