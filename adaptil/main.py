import os
import gc
import json
import torch
import numpy as np
import random
import argparse
from tqdm.notebook import tqdm
import pandas as pd
from config import config
from dataset.dataset import create_loaders
from evaluation import evaluate
from utils import *
import pytorch_lightning as pl
from transformers import AutoTokenizer
from Trainer import LightningModel
import warnings
warnings.filterwarnings('ignore')

if __name__=="__main__":

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)



    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    # parser.add_argument("-t", "--Task", help="'sa' for sentiment analysis, 'mnli' for multi_nli")



    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="model to train")
    parser.add_argument("--task", type=str, default="amazon_sa",
                        help="model to train")

    args = parser.parse_args()
    
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    device = torch.device("cpu")


    # os.environ["TOKENIZERS_PARALLELISM"] = 'false'


    results = {}

    # model_list = config["models"]

    # model_src_trg = []
    # accuracy_scores = []
    # f1_scores = []

    # for model_name in tqdm.tqdm(model_list):
    task = args.__dict__['task']#"sa" #args.Task  # define your task here
    model_name = args.__dict__['model']#'bert-base-uncased'# 'bert-base-uncased'

    PATH = os.path.join(os.getcwd(), "outputs", task, model_name)
    os.makedirs(PATH, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, usefast=True, use_lower_case=True)
    loaders = create_loaders(task=task, tokenizer=tokenizer)

    for source in tqdm(loaders):


        lm = LightningModel(model_name=model_name, task_config=config['tasks'][task])

        # create the checkpoint path
        MODEL_PATH = os.path.join(PATH, source)  # to load model

        os.makedirs(MODEL_PATH, exist_ok=True)

        run_name = task+"$"+model_name +"$"+source

        trainer = create_trainer(callback_config=config['callback_config'], path=PATH, run_name=run_name)

        train_loader, valid_loader, test_loader = loaders[source]['train'], loaders[source]['valid'], loaders[source]['valid']

        trainer.fit(lm, train_loader, valid_loader)

        # load best checkpoint
        # lm.load_from_checkpoint(MODEL_PATH)

        # this loads the best checkpoint
        trainer.test(
            model=lm,
            test_dataloaders=test_loader,
            verbose=False,
            ckpt_path="best",
        )
        

        # evaluate the best model on target domains
        for target in tqdm(loaders):


            f1, accuracy, cr  = evaluate(model=lm, loader=loaders[target]['valid'], device=device)
            
            
            results = update_results(
                task=task, 
                model_name=model_name,
                source=source,target=target, f1=f1, accuracy=accuracy, results=results
            )
                

            # model_src_trg.append((model_name, source, target))
            # accuracy_scores.append(accuracy)
            # f1_scores.append(f1)
                
            # save the classification report
            with open(os.path.join(MODEL_PATH, target+".txt"), "w") as file:
                file.write(cr)
            

        # delete model
        del lm
        gc.collect()
        torch.cuda.empty_cache()
        # break

    
    # save the results into json file at outputs/
    with open(os.path.join(PATH,  "results.json"), "w") as file:
        json.dump(results, file)

    print("Run Succesfully")
    gc.collect()


    

