import torch
from torch.utils.data import Dataset

class HANSDataset(Dataset):

    def __init__(self, dataset, tokenizer, max_len=128):

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
    

    def __len__(self):

        return len(self.dataset)


    def __getitem__(self, index):

        premise = self.dataset[index]['premise']
        hypothesis = self.dataset[index]['hypothesis']
        label = self.dataset[index]['label']
        template = self.dataset[index]['template']
        heuristic = self.dataset[index]['heuristic']

        input_encoding = self.tokenizer.encode_plus(
            text=[premise, hypothesis],
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
        )

        return {
            "input_ids": input_encoding['input_ids'].squeeze(),
            "attention_mask": input_encoding['attention_mask'].squeeze(),
            "label":torch.tensor([label], dtype=torch.long),
            "premise":premise,
            "hypothesis":hypothesis,
            "template":template,
            "heuristic":heuristic

        }