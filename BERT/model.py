# 출처
# Huggingface BERT : https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/bert/modeling_bert.py#L407
# BERT code 이해 : https://hyen4110.tistory.com/87

import math
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from transformers import BertTokenizer


# 1. BERT Embedding
class BERTEmbedding(nn.Module):
    def __init__(self, config):
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids, position_ids):
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# 2. BERT Self Attention
class BERTSelfAttention(nn.Module):
    def _init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, output_atttentions: bool):
        query_layer = self.attention_score(self.query(hidden_states))
        key_layer = self.attention_score(self.key(hidden_states))
        value_layer = self.attention_score(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size()[:-2] + (self.all_head_size,))

        outputs = (context_layer, attention_probs) if output_atttentions else (context_layer,)
        return outputs


# 3. BERT Attention
class BERTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BERTSelfAttention(config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        self_outputs = self.self(hidden_states)

        attention_output = self.dense(self_outputs[0])
        attention_output = self.dropout(attention_output)
        attention_output = self.LayerNorm(attention_output + hidden_states)

        outputs = (attention_output,) + self_outputs[1:]

        return outputs


# 4. BERT Layer
class BERTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BERTAttention(config)

    def forward(self, hidden_states):
        self_attention = self.attention(hidden_states)
        attention_output = self_attention[0]
        outputs = self_attention[1:]
        return {'attention_output': attention_output, 'outputs': outputs}


# 5. BERT Encoder
class BERTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BERTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, output_hidden_states):
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(hidden_states)
            hidden_states = layer_outputs[0]

        return {'hidden_states': all_hidden_states, 'last_hidden_state': hidden_states}


# 6. BERT Pooler
class BERTPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 7. BERT Model
class BERTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = BERTEncoder(config)
        self.embedding = BERTEmbedding(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, position_ids, token_type_ids):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        output_hidden_states = self.config.output_hidden_state

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids
        )

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return (sequence_output, pooled_output) + encoder_outputs[1:]


class BERTClassifier(L.LightningModule):
    def __init__(self, ):
        super().__init__()
        self.bert = BERTModel()
        self.classifier = nn.Linear()
        self.dropout = nn.Dropout()
