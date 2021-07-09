from transformers import AutoTokenizer
from dataset.loader import create_loaders
import argparse
import torch
from models.model import Model 
from evaluation import evaluate
from config import config


if __name__ == '__main__':

    device = torch.device("cpu")

    parser = argparse.ArgumentParser(description='debug this loaders, model and pipeline')

    parser.add_argument("--task", type=str, default="imdb_sst2_sa",
                        help="model to train")

    
    args = parser.parse_args()

    task = args.task

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


    loaders = create_loaders(task=task, tokenizer=tokenizer)

    model = Model(model_name="distilbert-base-uncased")
    for param in model.parameters():
        param.requires_grad = False



    f1, accuracy, cr, true_label, pred_label, pred_probs = evaluate(model=model, loader=loaders[config['tasks'][task]['training_domain']]['valid'], device=device)

    print(pred_probs)


    for domain in loaders:

        for set in loaders[domain]:

            print(f'domain = {domain} | set = {set} | len = {len(loaders[domain][set])}')





