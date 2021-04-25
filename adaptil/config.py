
config = {
    "tasks":{
        "sa":{
            "num_classes":2,
            "domains":["books", "dvd", "electronics", "kitchen_housewares"],
            # "domains":["books"],
            "lr":5e-3,
            "batch_size":32,
            "epochs":2,
            "average":"macro",

        },
        "mnli":{
            "num_classes":3,
            "domains":['government', 'telephone', 'fiction', 'travel', 'slate'],
            "mismatched_domains": ['letters', 'verbatim', 'facetoface', 'oup', 'nineeleven'],
            "lr":5e-3,
            "batch_size":32,
            "epochs":1,
            "average":"macro",
        },

    },
    "models":['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'distilroberta-base'],
    # "models":['distilbert-base-uncased'],

    "max_seq_length": 128,
    "num_workers":4,


    "callback_config":{
        "monitor":"val_accuracy",
        "min_delta":0.001,
        "patience":3,
        "precision":32,
        "project":"robust-representation-learning",
    }


}
