# utils
import torch

# data
from models.model import Model


# models
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AdamW

# training and evaluation
import wandb
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class LightningModel(pl.LightningModule):

    def __init__(self, model_name, task, config):

        super(LightningModel, self).__init__()

        self.config = config


        self.model = Model(model_name=model_name, num_classes=config['tasks'][task]['num_classes'])


    def forward(self, input_ids, attention_mask=None):
        logits, pooler  = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return logits, pooler

    def configure_optimizers(self):
        return optim.AdamW(params=self.parameters(), lr=self.config['training']['lr'])

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['label'].squeeze()
        logits, _ = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(logits, targets)

        acc = accuracy_score(targets.cpu(), logits.argmax(dim=1).cpu())

        f1 = f1_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['training']['average'])
        wandb.log({"loss":loss, "accuracy":acc, "f1_score":f1})
        return {"loss":loss, "accuracy":acc, "f1_score":f1}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['label'].squeeze()
        logits, _ = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(logits, targets)
        acc = accuracy_score(targets.cpu(), logits.argmax(dim=1).cpu())
        f1 = f1_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['training']['average'])
        precision = precision_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['training']['average'])
        recall = recall_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['training']['average'])
        return {"val_loss":loss, "val_accuracy":torch.tensor([acc]), "val_f1":torch.tensor([f1]), "val_precision":torch.tensor([precision]), "val_recall":torch.tensor([recall])}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        avg_precision = torch.stack([x['val_precision'] for x in outputs]).mean()
        avg_recall = torch.stack([x['val_recall'] for x in outputs]).mean()
        wandb.log({"val_loss":avg_loss, "val_accuracy":avg_acc, "val_f1":avg_f1, "val_precision":avg_precision, "val_recall":avg_recall})
        return {"val_loss":avg_loss, "val_accuracy":avg_acc, "val_f1":avg_f1, "val_precision":avg_precision, "val_recall":avg_recall}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['label'].squeeze()
        logits, _ = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(logits, targets)
        acc = accuracy_score(targets.cpu(), logits.argmax(dim=1).cpu())
        f1 = f1_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['training']['average'])
        precision = precision_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['training']['average'])
        recall = recall_score(targets.cpu(), logits.argmax(dim=1).cpu(), average=self.config['training']['average'])
        return {"test_loss":loss, "test_precision":torch.tensor([precision]), "test_recall":torch.tensor([recall]), "test_accuracy":torch.tensor([acc]), "test_f1":torch.tensor([f1])}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()
        avg_precision = torch.stack([x['test_precision'] for x in outputs]).mean()
        avg_recall = torch.stack([x['test_recall'] for x in outputs]).mean()
        return {"test_loss":avg_loss, "test_precision":avg_precision, "test_recall":avg_recall, "test_acc":avg_acc, "test_f1":avg_f1}