import torch.nn as nn
from transformers import AutoModel, AutoConfig

class Model(nn.Module):

    def __init__(self, model_name,freeze=0, num_classes=2):
        super().__init__()
        
        # freeze 0 means no freezing otherwise there will be portion given to freeze during pretraining

        # pretrained transformer model as base
        self.base = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        
        
        
        # freeze layers
        if freeze!=0:
            if "distil" in model_name:
                freeze_idx = int(freeze*6)
            else:
                freeze_idx = int(freeze*12)
                
            # surprisingly encoder in distilbert is named as Transformer
            if model_name=="distilbert-base-uncased":
                modules = [self.base.embeddings, *self.base.transformer.layer[:freeze_idx]] #Replace 5 by what you want
            else:
                modules = [self.base.embeddings, *self.base.encoder.layer[:freeze_idx]] #Replace 5 by what you want

            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False


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


