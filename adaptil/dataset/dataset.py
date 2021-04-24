import os
import torch

from config import config

from datasets import load_dataset
from torch.utils.data import DataLoader


def sa_loaders(tokenizer):
    '''
    file_name              train val

    kitchen_housewares.csv  1598 400
    electronics.csv         1596 399
    books.csv               1590 397
    dvd.csv                 1582 396

    For training, 1582 samples are used across all domains
    For validation, 396 samples are used across all domains
    '''

    domain_loaders = {}
    domain_list = config['tasks']['sa']['domains']

    for domain in domain_list:  # iterate through all domains and save loaders

        path_to_csv = os.path.join(os.getcwd(), "data", "amazon-review", domain+".csv")

        train_dataset = load_dataset('csv', data_files=path_to_csv, split='train[:80%]')
        val_dataset = load_dataset('csv', data_files=path_to_csv, split='train[80%:]')

        # train_dataset['label'] = train_dataset['sentiment']
        # valid_dataset['label'] = train_dataset['sentiment']


        # # does not work with datasets version that have unless I install from git
        train_dataset = train_dataset.rename_column('sentiment', 'label') 
        val_dataset = val_dataset.rename_column('sentiment', 'label')

        encoded_train_dataset = train_dataset.map(lambda x: tokenizer(x['review_text'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
        encoded_val_dataset = val_dataset.map(lambda x: tokenizer(x['review_text'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)

        encoded_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        encoded_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        train_data_loader = torch.utils.data.DataLoader(dataset = encoded_train_dataset, batch_size=config['tasks']['sa']["batch_size"], shuffle=True, num_workers=4)
        val_data_loader = torch.utils.data.DataLoader(dataset = encoded_val_dataset, batch_size=config['tasks']['sa']["batch_size"], shuffle=False, num_workers=4)

        domain_loaders[domain] = {
            "train": train_data_loader,
            "valid": val_data_loader
        }

    return domain_loaders


def mnli_loaders(tokenizer):

    dataset = load_dataset("multi_nli")
    dataset.remove_columns(['pairID', 'promptID', 'premise_binary_parse', 'premise_parse', 'hypothesis_binary_parse', 'hypothesis_parse'])

    domain_loaders = {}
    domain_list = config['tasks']['mnli']['domains']

    for domain in domain_list:  # iterate through all domains and save loaders

        domain_dataset = dataset.filter(lambda example: example['genre'] == domain)
        domain_dataset['train'] = domain_dataset['train'].select(range(77306))  # same for training across all domains
        domain_dataset['validation_matched'] = domain_dataset['validation_matched'].select(range(1945)) # same for validation across all domains

        encoded_domain_dataset = domain_dataset.map(lambda x: tokenizer(x['premise'], x['hypothesis'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
        encoded_domain_dataset['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        encoded_domain_dataset['validation_matched'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        train_data_loader = torch.utils.data.DataLoader(dataset = encoded_domain_dataset['train'], batch_size=config['tasks']['mnli']["batch_size"], shuffle=True, num_workers=4)
        val_data_loader = torch.utils.data.DataLoader(dataset = encoded_domain_dataset['validation_matched'], batch_size=config['tasks']['mnli']["batch_size"], shuffle=False, num_workers=4)

        domain_loaders[domain] = {
            "train": train_data_loader,
            "valid": val_data_loader
        }

    mismatched_domain_list = config['tasks']['mnli']['mismatched_domains']

    for domain in mismatched_domain_list:  # iterate through all domains and save loaders

        domain_dataset = dataset.filter(lambda example: example['genre'] == domain)
        domain_dataset['validation_mismatched'] = domain_dataset['validation_mismatched'].select(range(1945)) # same for validation across all domains

        encoded_domain_dataset = domain_dataset.map(lambda x: tokenizer(x['premise'], x['hypothesis'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
        encoded_domain_dataset['validation_mismatched'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        val_data_loader = torch.utils.data.DataLoader(dataset = encoded_domain_dataset['validation_mismatched'], batch_size=config['tasks']['mnli']["batch_size"], shuffle=False, num_workers=4)

        domain_loaders[domain] = {
            "valid": val_data_loader
        }

    return domain_loaders


def create_loaders(task, tokenizer):
    if task == "sa":
        return sa_loaders(tokenizer=tokenizer)
    else:
        return mnli_loaders(tokenizer=tokenizer)



