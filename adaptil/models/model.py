import torch.nn as nn
from transformers import AutoModel, AutoConfig

class Model(nn.Module):

    def __init__(self, model_name, num_classes=2):

        super().__init__()
        
        # self.config = AutoConfig.from_pretrained(model_name)

        # self.base_model = AutoModel.from_pretrained(model_name, config=self.config)

        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(0.10),
        #     nn.Linear(in_features=self.config.hidden_size, out_features=num_classes),  # change it to config.num_labels
        # )
        super().__init__()
        

        # pretrained transformer model as base
        self.base = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)


        # nn classifier on top of base model
        self.classfier = nn.Sequential(*[
            nn.Linear(in_features=768, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=num_classes),
        ])

    def forward(self, input_ids, attention_mask=None):

        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        
        
        # pooler.shape = [batch_size, hidden_size]

        logits = self.classfier(pooler)


        return logits


