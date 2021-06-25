
import torch
from datasets import load_dataset, concatenate_datasets



def mnli_datasets(config):
    pass

def mnli_loaders(dataset, config, tokenizer):

    domains = config['domains']

    train = 'train'
    test = 'validation_matched'

    # load the dataset
    dataset = load_dataset("multi_nli")
    dataset = dataset.remove_columns(['pairID', 'promptID', 'premise_binary_parse', 'premise_parse', 'hypothesis_binary_parse', 'hypothesis_parse']) # remove the unrelated fields

    labels = set(dataset['train']['label'])

    # which label has least number of samples in train data as well as (valid)validation data in all domains
    train_label_dist = 25000 # manually checked 
    test_label_dist = 600 # manually checked 
 

    domain_dsets = {}
    for domain in domains:

        domain_dsets[domain] = {
            "train":[]
        }
        domain_dsets[domain].update({"test":[]})

    
    for label in labels:

        for domain in domains:

            # filter by domain and by label # later all labels will be merged
            train = dataset['train'].filter(lambda example:example['genre']==domain).filter(lambda example:example['label']==label).select(range(train_label_dist))
            test = dataset['validation_matched'].filter(lambda example:example['genre']==domain).filter(lambda example:example['label']==label).select(range(test_label_dist))

            domain_dsets[domain]['train'].append(train)
            domain_dsets[domain]['test'].append(test)
    

    # concatenate the dataset and shuffle them
    # before it would be list of datasets and after it will be single dataset
    for domain in domains:

        # concate class label datasets
        domain_dsets[domain]['train'] = concatenate_datasets(dsets=domain_dsets[domain]['train']).shuffle()
        domain_dsets[domain]['test'] = concatenate_datasets(dsets=domain_dsets[domain]['test']).shuffle()


        # # tokenize 
        domain_dsets[domain]['train'] = domain_dsets[domain]['train'].map(lambda x: tokenizer(x['premise'], x['hypothesis'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
        domain_dsets[domain]['test'] = domain_dsets[domain]['test'].map(lambda x: tokenizer(x['premise'], x['hypothesis'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
        


        # change the dtype 
        domain_dsets[domain]['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        domain_dsets[domain]['test'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        # create dataloaders
        domain_dsets[domain]['train'] = torch.utils.data.DataLoader(dataset = domain_dsets[domain]['train'], batch_size=config["batch_size"], shuffle=True, num_workers=4)
        domain_dsets[domain]['test'] = torch.utils.data.DataLoader(dataset = domain_dsets[domain]['test'], batch_size=config["batch_size"], shuffle=True, num_workers=4)
        domain_dsets[domain]['valid'] = domain_dsets[domain]['test'] # validation and test will be same
        
        


    # why the hell I am doing this?
    

    loaders = domain_dsets
    

    return loaders