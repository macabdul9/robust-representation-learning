import torch.nn as nn
from transformers import AutoModel

class Model(nn.Module):

    def __init__(self, model_name, num_classes=2):
        super().__init__()
        
        # freeze 0 means no freezing otherwise there will be portion given to freeze during pretraining

        # pretrained transformer model as base
        self.base = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        

        # nn classifier on top of base model
        self.classifier = nn.Sequential(*[
            nn.Linear(in_features=768, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=num_classes),
        ])

    def forward(self, input_ids, attention_mask=None):

        # last hidden states
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        
        # cls token from last hidden states
        pooler = outputs[0][:, 0]

        # pass it to nn classifier
        logits = self.classifier(pooler)


        return logits


