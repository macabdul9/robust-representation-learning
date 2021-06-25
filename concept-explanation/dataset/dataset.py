from .mnli import mnli_datasets
from .imdb_sst2 import imdb_sst2_datasets
from .paraphrase import paraphrase_datasets
from config import config



def create_datasets(task):

    if task == "imdb_sst2_sa":
        return imdb_sst2_datasets(config=config['tasks'][task])
    
    elif task == "mnli":
        return mnli_datasets(config=config['tasks'][task])
    
    elif task=="paraphrase":
        return paraphrase_datasets(config=config['tasks'][task])