from transformers import AutoTokenizer
from dataset.imdb_sst2 import imdb_sst2_loaders

from config import config

if __name__=="__main__":
    
    
    task = "imdb_sst2_sa"
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")
    
    loaders = imdb_sst2_loaders(config=config['tasks'][task], tokenizer=tokenizer)
    
    for domain in loaders:
        print(domain, len(loaders[domain]['train']), len(loaders[domain]['test']), len(loaders[domain]['valid']))
        
        
    print("DataLoader Created!")
    