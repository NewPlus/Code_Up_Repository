import math
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
import torchmetrics


# 1. BERT Embedding
class BERTEmbedding(nn.Module):
    def __init__(self, config):
        super(BERTEmbedding, self).__init__()

        self.config = config
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids):
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# 2. BERT Self Attention
class BERTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, output_atttentions: bool):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.reshape(
            context_layer.shape[0], context_layer.shape[1], context_layer.shape[2], -1
        )
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(
            context_layer.size()[:-1] + (self.all_head_size,)
        )

        outputs = (
            (context_layer, attention_probs) if output_atttentions else (context_layer,)
        )
        return outputs


# 3. BERT Attention
class BERTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self = BERTSelfAttention(config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        self_outputs = self.self(hidden_states, False)

        attention_output = self.dense(self_outputs[0])
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)

        outputs = (attention_output,) + self_outputs[1:]

        return outputs


# 4. BERT Layer
class BERTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = BERTAttention(config)

    def forward(self, hidden_states):
        self_attention = self.attention(hidden_states)
        attention_output = self_attention[0]
        outputs = self_attention[1:]
        return {"attention_output": attention_output, "outputs": outputs}


# 5. BERT Encoder
class BERTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BERTLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states):
        for i, layer in enumerate(self.layer):
            layer_outputs = layer(hidden_states)
            hidden_states = layer_outputs["attention_output"]

        return {"hidden_states": layer_outputs, "last_hidden_state": hidden_states}


# 6. BERT Pooler
class BERTPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = torch.mean(pooled_output, dim=1)
        return pooled_output


# 7. BERT Model
class BERTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = BERTEncoder(config)
        self.embedding = BERTEmbedding(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        embedding_output = self.embedding(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        encoder_outputs = self.encoder(embedding_output)
        # print(encoder_outputs['hidden_states']['attention_output'])
        sequence_output = encoder_outputs["hidden_states"]["attention_output"]
        pooled_output = self.pooler(sequence_output)

        return pooled_output


class BERTClassifier(L.LightningModule):
    def __init__(self, config, num_classes, dropout_rate, learning_rate):
        super().__init__()
        self.bert = BERTModel(config=config)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.lr = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.f1_score = torchmetrics.F1Score(task="binary")

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.bert(input_ids, attention_mask, token_type_ids)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

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
            batch["review"]["input_ids"],
            batch["review"]["attention_mask"],
            batch["review"]["token_type_ids"],
            batch["sentiment"],
        )
        logits = self(input_ids, attention_mask, token_type_ids=token_type_ids)
        labels = labels.view(-1, 1).to(torch.float)
        logits = logits.to(torch.float)
        # print(labels, logits)
        loss = self.loss(logits, labels)
        pred = torch.argmax(logits, dim=-1).unsqueeze(1)
        return loss, logits, pred, labels

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
