import os
import gc
import json
import torch
import numpy as np
import pandas as pd
import random
import argparse
from tqdm import tqdm
import pandas as pd
from config import config
from utils import seed, create_trainer, update_results
import pytorch_lightning as pl
from evaluation import evaluate
from dataset.loader import create_loaders
from dataset.dataset import create_datasets
from transformers import AutoTokenizer
from Trainer import LightningModel


import warnings
warnings.filterwarnings('ignore')



    
if __name__=="__main__":

    seed(42)

    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--task", type=str, default="amazon_sa",
                        help="model to train")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch_size for training the model ")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="learning rate for model training")


    args = parser.parse_args()
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # device =  torch.device("cpu")
    

    results = {}

    task = args.__dict__['task']#"sa" #args.Task  # define your task here


    PATH = os.path.join(os.getcwd(), "outputs", task)
    os.makedirs(PATH, exist_ok=True)




    datasets = create_datasets(task=task)

    # create empty dataframes to save results
    results_dfs = {}
    for domain in datasets:
        results_dfs[domain] = pd.DataFrame()

    # empty dictionary to save results
    results = {}



    # train all the model in model list
    for i, model_name in enumerate(config['models']):

        # useful to save the results with first word of hf model name
        m = model_name.split("-")[0]


        # create tokenizers and loaders
        tokenizer = AutoTokenizer.from_pretrained(model_name, usefast=True, use_lower_case=True)
        loaders = create_loaders(
            dataset=datasets,
            task=task,
            tokenizer=tokenizer
        )

        # create lighning model
        lm = LightningModel(model_name=model_name, task=task, config=config)


        # path to save the trained models checkpoints
        MODEL_PATH = os.path.join(PATH, model_name)  # to load model

        os.makedirs(MODEL_PATH, exist_ok=True)


        trainer = create_trainer(config=config, run_name=task + "-" + m, ckpt_path=MODEL_PATH)



        trainer.fit(
            model=lm,
            train_dataloader=loaders[config['tasks'][task]['training_domain']]['train'],
            val_dataloaders=loaders[config['tasks'][task]['training_domain']]['valid'],
        )


        # # load the trained model
        # lm.load_from_checkpoint(MODEL_PATH)

        # this loads the best checkpoint
        trainer.test(
            model=lm,
            test_dataloaders=loaders[config['tasks'][task]['training_domain']]['test'],
            verbose=True,
            ckpt_path="best",
        )


        for domain in loaders:

            gt = []
            input_texts = []
            for batch in loaders[domain]['test']:
                gt += batch['label'].cpu().tolist()
                for each in batch['input_ids']:
                    input_texts.append(tokenizer.decode(
                            token_ids = each.tolist(),
                            skip_special_tokens=True
                        )
                    )
                # break

            results_dfs[domain]['text'] = input_texts
            results_dfs[domain]['ground_truth'] = gt

            

            # ideally order of gt and true label should be same

            f1, accuracy, cr, true_label, pred_label, pred_probs = evaluate(model=lm, loader=loaders[domain]['test'], device=device)


            results = update_results(
                task=task, 
                model_name=m,
                source=config['tasks'][task]['training_domain'],
                target=domain, 
                f1=f1, 
                accuracy=accuracy, 
                results=results
            )

            # results_dfs[domain][m+"_pred"] = pred_label
            # results_dfs[domain][m+"_prob"] = pred_probs

            if results_dfs[domain]['ground_truth'].values.tolist()==true_label:
                results_dfs[domain][m+"_pred"] = pred_label
                results_dfs[domain][m+"_prob"] = pred_probs
            else:
                raise ValueError("order has changed!")

            
        # if model_name == "bert-base-uncased" and i > 1:




        del lm 
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

        # break



    # save the results into json file at outputs/
    with open(os.path.join(PATH,  "results.json"), "w") as file:
        json.dump(results, file)
    
    for domain in results_dfs.keys():
        results_dfs[domain].to_csv(os.path.join(PATH, domain+".csv"), index=False)




    # get the mean probability from dataframes



    del results
    del results_dfs
    gc.collect()

    

