import os
import torch

from config import config

from datasets import load_dataset
from torch.utils.data import DataLoader
from .SADataset import SADataset

from .imdb_sst2 import imdb_sst2_loaders
from .mnli import mnli_loaders
from  .qqp_paws import  qqp_paws_loaders


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
        
        
        encoded_train_dataset = SADataset(tokenizer=tokenizer, file_name=os.path.join(config['tasks']['sa']['dataset_path'], domain, "train.csv"))
        encoded_val_dataset = SADataset(tokenizer=tokenizer, file_name=os.path.join(config['tasks']['sa']['dataset_path'], domain, "valid.csv"))
        # train = SADataset(tokenizer=tokenizer, file_name=os.path.join(config['dataset_path'], domain, "train.csv"))
        
        # path_to_csv = os.path.join(os.getcwd(), "data", "amazon-review", domain+".csv")

        # train_dataset = load_dataset('csv', data_files=path_to_csv, split='train[:80%]')
        # val_dataset = load_dataset('csv', data_files=path_to_csv, split='train[80%:]')
        
        # train_dataset = train_dataset.rename_column('sentiment', 'label') 
        # val_dataset = val_dataset.rename_column('sentiment', 'label')

        # encoded_train_dataset = train_dataset.map(lambda x: tokenizer(x['review_text'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
        # encoded_val_dataset = val_dataset.map(lambda x: tokenizer(x['review_text'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)

        # encoded_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        # encoded_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        train_data_loader = torch.utils.data.DataLoader(dataset = encoded_train_dataset, batch_size=config['tasks']['sa']["batch_size"], shuffle=True, num_workers=4)
        val_data_loader = torch.utils.data.DataLoader(dataset = encoded_val_dataset, batch_size=config['tasks']['sa']["batch_size"], shuffle=False, num_workers=4)

        domain_loaders[domain] = {
            "train": train_data_loader,
            "valid": val_data_loader
        }

    return domain_loaders

# def sa_loaders(tokenizer):
#     '''
#     file_name              train val

#     kitchen_housewares.csv  1598 400
#     electronics.csv         1596 399
#     books.csv               1590 397
#     dvd.csv                 1582 396

#     For training, 1582 samples are used across all domains
#     For validation, 396 samples are used across all domains
#     '''

#     domain_loaders = {}
#     domain_list = config['tasks']['sa']['domains']

#     for domain in domain_list:  # iterate through all domains and save loaders

#         path_to_train_csv = os.path.join(os.getcwd(), "data", "amazon-review", domain, "train.csv")
#         path_to_valid_csv = os.path.join(os.getcwd(), "data", "amazon-review", domain, "valid.csv")
#         path_to_test_csv = os.path.join(os.getcwd(), "data", "amazon-review", domain, "test.csv")

#         train_dataset = load_dataset('csv', data_files=path_to_train_csv)['train']
#         valid_dataset = load_dataset('csv', data_files=path_to_valid_csv)['train']
#         test_dataset = load_dataset('csv', data_files=path_to_test_csv)['train']


#         # # does not work with datasets version that have unless I install from git
#         train_dataset = train_dataset.rename_column('sentiment', 'label') 
#         valid_dataset = valid_dataset.rename_column('sentiment', 'label')
#         test_dataset = test_dataset.rename_column('sentiment', 'label')

#         encoded_train_dataset = train_dataset.map(lambda x: tokenizer(x['review_text'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
#         encoded_valid_dataset = valid_dataset.map(lambda x: tokenizer(x['review_text'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
#         encoded_test_dataset = test_dataset.map(lambda x: tokenizer(x['review_text'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)

#         encoded_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
#         encoded_valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
#         encoded_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

#         train_data_loader = torch.utils.data.DataLoader(dataset = encoded_train_dataset, batch_size=config['tasks']['sa']["batch_size"], shuffle=True, num_workers=4)
#         valid_data_loader = torch.utils.data.DataLoader(dataset = encoded_valid_dataset, batch_size=config['tasks']['sa']["batch_size"], shuffle=False, num_workers=4)
#         test_data_loader = torch.utils.data.DataLoader(dataset = encoded_test_dataset, batch_size=config['tasks']['sa']["batch_size"], shuffle=False, num_workers=4)

#         domain_loaders[domain] = {
#             "train": train_data_loader,
#             "valid": valid_data_loader,
#             "test": test_data_loader
#         }

#     return domain_loaders


# def mnli_loaders(tokenizer):

#     dataset = load_dataset("multi_nli")
#     dataset.remove_columns(['pairID', 'promptID', 'premise_binary_parse', 'premise_parse', 'hypothesis_binary_parse', 'hypothesis_parse'])

#     domain_loaders = {}
#     domain_list = config['tasks']['mnli']['domains']

#     for domain in domain_list:  # iterate through all domains and save loaders

#         domain_dataset = dataset.filter(lambda example: example['genre'] == domain)
#         domain_dataset['train'] = domain_dataset['train'].select(range(50))  # this is actual value 77306# same for training across all domains
#         domain_dataset['validation_matched'] = domain_dataset['validation_matched'].select(range(500)) # 1945 # same for validation across all domains

#         encoded_domain_dataset = domain_dataset.map(lambda x: tokenizer(x['premise'], x['hypothesis'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
#         encoded_domain_dataset['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
#         encoded_domain_dataset['validation_matched'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

#         train_data_loader = torch.utils.data.DataLoader(dataset = encoded_domain_dataset['train'], batch_size=config['tasks']['mnli']["batch_size"], shuffle=True, num_workers=4)
#         val_data_loader = torch.utils.data.DataLoader(dataset = encoded_domain_dataset['validation_matched'], batch_size=config['tasks']['mnli']["batch_size"], shuffle=False, num_workers=4)

#         domain_loaders[domain] = {
#             "train": train_data_loader,
#             "valid": val_data_loader
#         }

#     # # We are not going to to this
#     # mismatched_domain_list = config['tasks']['mnli']['mismatched_domains']

#     # for domain in mismatched_domain_list:  # iterate through all domains and save loaders

#     #     domain_dataset = dataset.filter(lambda example: example['genre'] == domain)
#     #     domain_dataset['validation_mismatched'] = domain_dataset['validation_mismatched'].select(range(1945)) # same for validation across all domains

#     #     encoded_domain_dataset = domain_dataset.map(lambda x: tokenizer(x['premise'], x['hypothesis'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
#     #     encoded_domain_dataset['validation_mismatched'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

#     #     val_data_loader = torch.utils.data.DataLoader(dataset = encoded_domain_dataset['validation_mismatched'], batch_size=config['tasks']['mnli']["batch_size"], shuffle=False, num_workers=4)

#     #     domain_loaders[domain] = {
#     #         "valid": val_data_loader
#     #     }

#     return domain_loaders


def create_loaders(task, tokenizer):
    
    if task == "amazon_sa":
        return sa_loaders(tokenizer=tokenizer)
    
    elif task == "imdb_sst2_sa":
        return imdb_sst2_loaders(config=config['tasks'][task], tokenizer=tokenizer)
    
    elif task == "mnli":
        return mnli_loaders(config=config['tasks'][task], tokenizer=tokenizer)
    
    elif task=="paraphrase":
        return qqp_paws_loaders(config=config['tasks'][task], tokenizer=tokenizer)
        



