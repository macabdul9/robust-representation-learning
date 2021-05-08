
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
            "batch_size":2, # 2 for test actul is 8, # large sequence length hence smaller batch size
            "epochs":1, # 1 for testing actual is 10,
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
            "domains":['qqp', 'paws'],
            "lr":2e-5,
            "batch_size":32,
            "epochs":1,
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


    "callback_config":{
        "monitor":"val_accuracy",
        "min_delta":0.001,
        "patience":2,
        "precision":32,
        "project":"robust-representation-learning",
    }


}
