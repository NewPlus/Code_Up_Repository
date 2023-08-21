import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import torchmetrics
from transformers import LlamaForCausalLM
from peft import LoraConfig, get_peft_model


class LlamaClassifier(L.LightningModule):
    def __init__(self, config, pretrain_name, num_classes, learning_rate):
        super().__init__()
        model_path = "./hf_llama2_weight_13b"
        model = LlamaForCausalLM.from_pretrained(model_path)
        # print(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        # print(config.vocab_size)
        self.llama2 = get_peft_model(model, lora_config)
        self.classifier = nn.Linear(config.vocab_size, 2)
        self.lr = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.f1_score = torchmetrics.F1Score(task="binary")

    def forward(self, **kwargs):
        x = self.llama2(**kwargs)
        logits = self.classifier(x.logits)
        return logits

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._common_step(batch, batch_idx)
        self.log_dict(
            {"train_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        loss, logits, pred, labels = self._common_step(batch, batch_idx)
        f1_score = self.f1_score(pred, labels)
        self.log_dict(
            {"val_loss": loss, "val_f1_score": f1_score},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, pred, labels = self._common_step(batch, batch_idx)
        f1_score = self.f1_score(pred, labels)
        self.log_dict({"test_loss": loss, "test_f1_score": f1_score})
        return loss

    def _common_step(self, batch, batch_idx):
        print(batch)
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
