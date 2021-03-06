
config = {
    "tasks":{
        "amazon_sa":{
            "dataset_path":"./data/amazon-review",
            "domains":["books", "dvd", "electronics", "kitchen_housewares"],
            # "domains":["books"],
            "num_classes":5, # instead of 2-class sentiment classification convert it to multiclass classfication
            "lr":2e-5, #    "lr":2e-5,
            "batch_size":32,
            "epochs":10,
            "average":"macro",
            "max_seq_length": 128,
            

        },
        "imdb_sst2_sa":{
            "domains":["imdb", "sst2"],
            "num_classes":2, # instead of 2-class sentiment classification convert it to multiclass classfication
            "lr":2e-5, #    "lr":2e-5,
            "batch_size":32, # 2 for test actul is 8, # large sequence length hence smaller batch size
            "epochs":5, # 1 for testing actual is 10,
            "average":"macro",
            "max_seq_length": 512, # imdb have lengthy reviews 
        },
        
        
        "mnli":{
            "num_classes":3,
            "domains":['government', 'telephone', 'fiction', 'travel', 'slate'],
            "lr":2e-5,
            "batch_size":32,
            "epochs":5,
            "average":"macro",
            "max_seq_length": 128,
        },
        
        "paraphrase":{
            "num_classes":2,
            "domains":['qqp', 'paws'], # "mrpc"
            "lr":2e-5,
            "batch_size":128, # actual 32, 128 for representation analysis
            "epochs":10,
            "average":"macro",
            "max_seq_length": 256,
        },
        
        "toxic-comments":{
            "num_classes":2,
            "domains":['male',
                        'female',
                        'LGBTQ',
                        'christian',
                        'muslim',
                        'other_religions',
                        'black',
                        'white'],
            "batch_size":32,
            
        }

    },
    "models":['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'distilroberta-base'],
    # "models":['distilbert-base-uncased'],

    # "max_seq_length": 128, # moved to respective task section
    "num_workers":4,
    
    'freeze':1/2, # 1/2 means half of total(6 if there are 12, 3 if there are 6), 1/3 means one third(4 if 12 2 if 6)


    "callback_config":{
        "monitor":"val_accuracy",
        "min_delta":0.001,
        "patience":3,
        "precision":32,
        "project":"robust-representation-learning",
    }


}
