
config = {
    "tasks":{
        "imdb_sst2_sa":{
            "domains":["imdb", "sst2"],
            "training_domain":"sst2",
            "num_classes":2, # instead of 2-class sentiment classification convert it to multiclass classfication
            "lr":2e-5, #    "lr":2e-5,
            "batch_size":32, # 2 for test actul is 8, # large sequence length hence smaller batch size
            "epochs":5, # 1 for testing actual is 5,
            "average":"macro",
            "max_seq_length": 256, # imdb have lengthy reviews 
        },
        
        
        "mnli":{
            "num_classes":3,
            "domains":['mnli', 'hans'],
            "training_domain":"mnli",
            "lr":2e-5,
            "batch_size":4,
            "epochs":5,
            "average":"macro",
            "max_seq_length": 128,
        },
        
        "paraphrase":{
            "num_classes":2,
            "domains":['qqp', 'paws'], # "mrpc"
            "training_domain":"qqp",
            "lr":2e-5,
            "batch_size":4, # actual 32, 128 for representation analysis
            "epochs":5,
            "average":"macro",
            "max_seq_length": 256,
        },


    },
    "models":['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'albert-base-v2'],


    # "max_seq_length": 128, # moved to respective task section
    "num_workers":4,

    "training":{
        "epochs":5,
        "lr":2e-5, 
        "average":"macro",
    },

    "callback":{
        "monitor":"val_accuracy",
        "min_delta":0.001,
        "patience":2,
        "precision":32,
        "project":"concept-explanation",
    }


}
