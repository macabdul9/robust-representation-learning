import torch
from datasets import load_dataset, concatenate_datasets
import numpy as np

def imdb_sst2_loaders(config, tokenizer):

    """
        We have to ensure that sample size as well label distribution remains uniform across the distribution and set. 
    """

    sst2 = load_dataset("toriving/sst2") # sst2 has train, valid and test. We're mering test and valid set into test set
    imdb = load_dataset("toriving/imdb")

    # try this if ood problem does not get fixed load_dataset("toriving/imdb")

    sst2_train = sst2['train'].shuffle()
    sst2_test = concatenate_datasets([sst2['validation'], sst2['test']]).shuffle()

    imdb_train = imdb['train'].shuffle()
    imdb_test = imdb['test'].shuffle()


    train_label_values = []
    test_label_values = []

    labels = np.unique(sst2_train['label']).tolist()

    for label in labels: # assuming that there's no label shift 

        # min number of samples of label in  both dataset  in train dataset
        train_min = min(len(sst2_train.filter(lambda example: example['label'] == int(label))), len(imdb_train.filter(lambda example: example['label'] == int(label))))
        
        train_label_values.append(train_min)

        # min number of samples of label in  both dataset  in test dataset
        test_min = min(len(sst2_test.filter(lambda example: example['label'] == int(label))), len(imdb_test.filter(lambda example: example['label'] == int(label))))
        test_label_values.append(test_min)
    

    train_label_dist = min(train_label_values)
    
    test_label_dist = min(test_label_values)


    ## 
    dsets = {

        "sst2":{
            "train":[],
            "test":[]
        },

        "imdb":{
            "train":[],
            "test":[]
        },
    }


    for label in labels:

        sst2_train_label = sst2_train.shuffle().filter(lambda example: example['label']==int(label)).select(range(train_label_dist))

        sst2_test_label = sst2_test.shuffle().filter(lambda example: example['label']==int(label)).select(range(test_label_dist))

        imdb_train_label = imdb_train.shuffle().filter(lambda example: example['label']==int(label)).select(range(train_label_dist))

        imdb_test_label = imdb_test.shuffle().filter(lambda example: example['label']==int(label)).select(range(test_label_dist))


        dsets['sst2']['train'].append(sst2_train_label)
        dsets['sst2']['test'].append(sst2_test_label)

        dsets['imdb']['train'].append(imdb_train_label)
        dsets['imdb']['test'].append(imdb_test_label)
        

    ## split the data based on sample distribution as well as label distribution
    sst2_train = concatenate_datasets(dsets=dsets['sst2']['train']).shuffle()
    sst2_test = concatenate_datasets(dsets=dsets['sst2']['test']).shuffle()

    imdb_train = concatenate_datasets(dsets=dsets['imdb']['train']).shuffle()
    imdb_test = concatenate_datasets(dsets=dsets['imdb']['test']).shuffle()


    # tokenize the dataset

    # this can be done with loop but who cares 

    # sst2
    # train
    sst2_train_tokenized = sst2_train.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
    sst2_train_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    # test
    sst2_test_tokenized = sst2_test.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
    sst2_test_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


    # imdb
    # train
    imdb_train_tokenized = imdb_train.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
    imdb_train_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # test
    imdb_test_tokenized = imdb_test.map(lambda x: tokenizer(x['text'], padding='max_length', truncation=True, max_length=config['max_seq_length']), batched=True)
    imdb_test_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])


    # print(sst2_train_tokenized, sst2_test_tokenized, imdb_train_tokenized, imdb_test_tokenized)

    sst2_trainloader = torch.utils.data.DataLoader(dataset = sst2_train_tokenized, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    sst2_testloader = torch.utils.data.DataLoader(dataset = sst2_test_tokenized, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    imdb_trainloader = torch.utils.data.DataLoader(dataset = imdb_train_tokenized, batch_size=config["batch_size"], shuffle=True, num_workers=4)

    imdb_testloader = torch.utils.data.DataLoader(dataset = imdb_test_tokenized, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    return {
        "sst2":{
            "train":sst2_trainloader,
            "valid":sst2_testloader,
            "test":sst2_testloader,
        },
        "imdb":{
            "train":imdb_trainloader,
            "valid":imdb_testloader,
            "test":imdb_testloader
        }
    }
