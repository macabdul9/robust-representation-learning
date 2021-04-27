import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class SADataset(Dataset):
    
    def __init__(self, tokenizer, file_name, text_field="review_text", label_field="sentiment", max_len=128):
        
        self.tokenizer = tokenizer
        
        self.data = load_dataset("csv", data_files=file_name)['train']
        
        
        self.text = self.data['review_text']
        self.label = self.data['sentiment']
        
        self.max_len = max_len
        
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        
        text = self.text[idx]
        label = self.label[idx]
        
        input_encoding = self.tokenizer.encode_plus(
            text=text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
        )
        
        return {
            "input_ids":input_encoding['input_ids'].squeeze(),
            "attention_mask":input_encoding['attention_mask'].squeeze(),
            "label":torch.tensor([label], dtype=torch.long)
        }