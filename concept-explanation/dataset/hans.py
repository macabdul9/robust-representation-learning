import torch
from datasets import load_dataset
from typing import Any
from .HANSDataset import HANSDataset


def hans_datasets(config):
    
    dataset = load_dataset("hans")

    train = dataset['train'].shuffle()
    valid = dataset['validation'].shuffle().select(range(10000))
    test = dataset['validation'].shuffle().select(range(10000))

    return {
        "hans":{
            "train":train,
            "valid":valid,
            "test":test,

        }
    }

def hans_loaders(dataset, config, tokenizer):
    
    loaders = {}

    
    for domain in dataset:
        loaders[domain]= {}

    for domain in dataset:
        for set in dataset[domain]:

            # tokenize current set and change the type
            tokenized = HANSDataset(dataset[domain][set], tokenizer=tokenizer, max_len=config['max_seq_length'])

            if set=="train":
                loaders[domain].update({
                    set:torch.utils.data.DataLoader(dataset = tokenized, batch_size=config["batch_size"], shuffle=True, num_workers=4)
                })
            else:
                loaders[domain].update({
                    set:torch.utils.data.DataLoader(dataset = tokenized, batch_size=config["batch_size"], shuffle=False, num_workers=4)
                })


    return loaders