import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import torchmetrics
from transformers import BertModel


class BERTClassifier(L.LightningModule):
    def __init__(self, config, pretrain_name, num_classes, learning_rate):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrain_name, config=config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.lr = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.f1_score = torchmetrics.F1Score(task="binary")

    def forward(self, **kargs):
        x = self.bert(**kargs)
        logits = x["last_hidden_state"][:, 0]
        logits = self.classifier(logits)
        return logits

    def training_step(self, batch, batch_idx):
        loss, logits, pred, labels = self._common_step(batch, batch_idx)
        f1_score = self.f1_score(pred, labels)
        self.log_dict(
            {"train_loss": loss, "train_f1_score": f1_score},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "logits": logits, "pred": pred}

    def validation_step(self, batch, batch_idx):
        loss, _ = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, pred, labels = self._common_step(batch, batch_idx)
        f1_score = self.f1_score(pred, labels)
        self.log_dict({"test_loss": loss, "test_f1_score": f1_score})
        return loss

    def _common_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["token_type_ids"],
            batch["label"],
        )

        logits = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pred = torch.argmax(logits, dim=-1).float()
        loss = self.loss(logits, labels)
        return loss, logits, pred, labels

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
