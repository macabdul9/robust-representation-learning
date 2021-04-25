import torch.nn as nn
from transformers import AutoModel, AutoConfig

class Model(nn.Module):

    def __init__(self, model_name, num_classes=2):

        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)

        self.base_model = AutoModel.from_pretrained(model_name, config=self.config)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.config.hidden_size, out_features=self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(in_features=self.config.hidden_size, out_features=num_classes),  # change it to config.num_labels
        )

    def forward(self, input_ids, attention_mask=None):

        last_hidden_state = self.base_model(input_ids, attention_mask)[0]
        cls_tokens = last_hidden_state[:, 0]
        logits = self.classifier(cls_tokens)
        return logits


