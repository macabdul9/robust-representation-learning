from .mnli import mnli_loaders
from .paraphrase import paraphrase_loaders
from .imdb_sst2 import imdb_sst2_loaders
from config import config


def create_loaders(dataset, task, tokenizer):
    
    if task == "imdb_sst2_sa":
        return imdb_sst2_loaders(dataset=dataset, config=config['tasks'][task], tokenizer=tokenizer)
    
    elif task == "mnli":
        return mnli_loaders(dataset=dataset, config=config['tasks'][task], tokenizer=tokenizer)
    
    elif task=="paraphrase":
        return paraphrase_loaders(dataset=dataset, config=config['tasks'][task], tokenizer=tokenizer)
    
    elif task=="hans":
        return paraphrase_loaders(dataset=dataset, config=config['tasks'][task], tokenizer=tokenizer)