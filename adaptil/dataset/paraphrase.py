import torch
from datasets import load_dataset, concatenate_datasets
import numpy as np

def paraphrase_loaders(config, tokenizer, max_len=256):

    domains = config['domains']

    # which label has least number of samples in train data as well as (test)validation data in all domains
    train_label_dist = 21829 # manually checked 
    test_label_dist =  7075 # manually checked 


    # we are  not going to take words greater than 256
    paws = load_dataset("paws", 'labeled_final')
    qqp = load_dataset("glue", 'qqp')


    # # If you want to filter the data based on length | no filtering in actul experiment
    for _, (paws_set, qqp_set) in enumerate(zip(paws.keys(), qqp.keys())):

        # # applying filter
        # paws[paws_set] = paws[paws_set].filter(lambda example : (len(example['sentence1'])+len(example['sentence2']))<=max_len)
        # qqp[qqp_set] = qqp[qqp_set].filter(lambda example : (len(example['question1'])+len(example['question2']))<=max_len)

        # both paws and qqp has difference names for 2 input sentences 
        # paws = (sentence1, sentence2) and qqp = (question1, question2) update qqp col to match paws

        qqp[qqp_set] = qqp[qqp_set].rename_column('question1', 'sentence1') 
        qqp[qqp_set] = qqp[qqp_set].rename_column('question2', 'sentence2')


    # # merge the validation and test of both datasets
    paws['test'] = concatenate_datasets(dsets=[paws['test'], paws['validation']])
    qqp['test'] = concatenate_datasets(dsets=[qqp['test'], qqp['validation']])

    datasets = {
        "paws":paws,
        "qqp":qqp
    }

    labels = list(set(qqp['train']['label']))

    domain_dsets = {}
    for domain in domains:

        domain_dsets[domain] = {
            "train":[]
        }
        domain_dsets[domain].update({"test":[]})

    # take equal numbe of samples for each domain for each label for each set
    for label in labels:
        
        for domain in domain_dsets:

            train = datasets[domain]['train'].filter(lambda example:example['label']==label).shuffle().select(range(train_label_dist))
            test = datasets[domain]['test'].filter(lambda example:example['label']==label).shuffle().select(range(test_label_dist))

            domain_dsets[domain]['train'].append(train)
            domain_dsets[domain]['test'].append(test)





    # concatenate the dataset and shuffle them
    # before it would be list of datasets and after it will be single dataset
    for domain in domains:

        # concate class label datasets
        domain_dsets[domain]['train'] = concatenate_datasets(dsets=domain_dsets[domain]['train']).shuffle()
        domain_dsets[domain]['test'] = concatenate_datasets(dsets=domain_dsets[domain]['test']).shuffle()


        # # tokenize 
        domain_dsets[domain]['train'] = domain_dsets[domain]['train'].map(lambda x: tokenizer(x['sentence1'], x['sentence2'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
        domain_dsets[domain]['test'] = domain_dsets[domain]['test'].map(lambda x: tokenizer(x['sentence1'], x['sentence2'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
        


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



