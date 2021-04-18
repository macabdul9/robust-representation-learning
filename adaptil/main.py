import os
import gc
import json
import torch
import argparse
import tqdm

from config import config
from dataset.dataset import create_loaders
from evaluation import evaluate
from utils import *

import pytorch_lightning as pl
from transformers import AutoTokenizer
from Trainer import LightningModel

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--Task", help="'sa' for sentiment analysis, 'mnli' for multi_nli")

    args = parser.parse_args()

    task = "sa" #args.Task  # define your task here
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    results = {}

    model_list = config["models"]

    for model_name in tqdm.tqdm(model_list):

        tokenizer = AutoTokenizer.from_pretrained(model_name, usefast=True, use_lower_case=True)
        loaders = create_loaders(task=task, tokenizer=tokenizer)

        for source in tqdm.tqdm(loaders):

            if(len(loaders[source])!=2):
                continue

            lm = LightningModel(model_name=model_name, task_config=config['tasks'][task])

            # create the checkpoint path
            PATH = os.path.join(os.getcwd(), "outputs", task, model_name)
            MODEL_PATH = os.path.join(PATH, source)  # to load model

            os.makedirs(PATH, exist_ok=True)
            os.makedirs(MODEL_PATH, exist_ok=True)

            run_name = task+"$"+model_name +"$"+source

            trainer = create_trainer(callback_config=config['callback_config'], path=PATH, run_name=run_name)

            train_loader, valid_loader = loaders[source]['train'], loaders[source]['valid']

            trainer.fit(lm, train_loader, valid_loader)

            # # load best checkpoint
            lm.load_from_checkpoint(MODEL_PATH)
            trainer.test(
                model=lm,
                test_dataloaders=valid_loader,
                verbose=False,
                ckpt_path="best",
            )
            

            # evaluate the best model on target domains
            for target in tqdm.tqdm(loaders):

                f1, accuracy, cr  = evaluate(model=lm, loader=loaders[target]['valid'], device=device)
                
                
                results = update_results(
                    task=task, 
                    model_name=model_name,
                    source=source,target=target, f1=f1, accuracy=accuracy, results=results
                )
                    
                    
                # save the classification report
                # with open(os.path.join(MODEL_PATH, target+".txt"), "w") as file:
                #     file.write(cr)
                

            # delete model
            del lm
            gc.collect()
            torch.cuda.empty_cache()


    # save the results into json file at outputs/
    with open(os.path.join(os.getcwd(), "outputs", "results.json"), "w") as file:
        json.dump(results, file)