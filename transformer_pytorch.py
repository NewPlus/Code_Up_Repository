###########################################################################
#                                                                         #
#              PyTorch Ligtning으로 Transformer 구현해보기                 #
#                                                                         #
###########################################################################

# - 이용환
# - from 'Attention is All You Need' & something else
# - PyTorch, einsum을 이번 기회에 사용해보고 익히기
# - PEP8 지켜보기!!
# - 구현할 것들
#   - Encoder Class : O
#   - Decoder Class : O
#   - Multi-Head Attention : O
#   - Multi-Head Masked Attention : O
#   - Multi-Head Cross Attention : O

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

# device = "cuda" if torch.cuda.is_available() else "cpu"


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_head: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_head
        self.head_dim = hidden_dim // num_head

        self.fcQ = nn.Linear(hidden_dim, self.head_dim * num_head)
        self.fcK = nn.Linear(hidden_dim, self.head_dim * num_head)
        self.fcV = nn.Linear(hidden_dim, self.head_dim * num_head)
        self.fcOut = nn.Linear(self.head_dim * num_head, hidden_dim)

    def forward(self, x, enc_k=None, enc_v=None, mask=None):
        # x : batch * Input length * hidden_dim

        # Cross Attention -> enc_k, enc_v is not None
        if (enc_k is not None) and (enc_v is not None):
            k, v = enc_k, enc_v
        else:
            k, v = x, x

        q = rearrange(self.fcQ(x), 'b s (n h) -> b n s h',
                      n=self.num_head)
        k = rearrange(self.fcK(k), 'b s (n h) -> b n s h',
                      n=self.num_head)
        v = rearrange(self.fcV(v), 'b s (n h) -> b n s h',
                      n=self.num_head)

        # q : batch * head * Input length * head_dim
        # k : batch * head * Input length * head_dim
        # v : batch * head * Input length * head_dim

        # scaled_dot_product : (Q (dot) K_T) / sqrt(d_k)
        scaled_dot_product = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        scaled_dot_product *= np.sqrt(self.head_dim)

        # for masked_attention
        if mask is not None:
            scaled_dot_product = torch.tril(scaled_dot_product)

        attn_score = torch.softmax(scaled_dot_product, dim=-1)
        out = torch.einsum('b h i j , b h j d -> b h i d', attn_score, v)
        out = rearrange(out, "b n sq h -> b sq (n h)")
        out = self.fcOut(out)

        return out


class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.layers(x)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_head: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.self_attn = MultiHeadAttention(hidden_dim, num_head)
        self.ff = FeedForwardLayer(hidden_dim)

    def forward(self, x):
        x += self.self_attn(x)
        x = self.norm(x)
        x += self.ff(x)
        x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_head: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.self_attn = MultiHeadAttention(hidden_dim, num_head)
        self.cross_attn = MultiHeadAttention(hidden_dim, num_head)
        self.ff = FeedForwardLayer(hidden_dim)

    def forward(self, x, enc_k, enc_v):
        x += self.self_attn(x)
        x = self.norm(x)
        x += self.cross_attn(x, enc_k=enc_k, enc_v=enc_v, mask=True)
        x = self.norm(x)
        x += self.ff(x)
        x = self.norm(x)
        # (batch_size, seq_len, emb_dim)
        return x


class Transformer(nn.Module):
    def __init__(self, hidden_dim: int,
                 num_head: int,
                 num_output: int,
                 num_dec: int):
        super().__init__()
        # num_dec x Encoder Layer, num_dec x Decoder Layer
        self.encoder = nn.ModuleList([EncoderLayer(hidden_dim, num_head)
                                      for _ in range(num_dec)])
        self.decoder = nn.ModuleList([DecoderLayer(hidden_dim, num_head)
                                      for _ in range(num_dec)])
        self.linear = nn.Linear(hidden_dim, num_output)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        enc_x = x
        for encoder in self.encoder:
            enc_x = encoder(enc_x)

        for decoder in self.decoder:
            x = decoder(x, enc_k=enc_x, enc_v=enc_x)

        x = self.linear(x)
        x = self.softmax(x)
        return x


# batch_size = 2 , seq_len = 512, emb_dim = 768 , 3 = (q,v,k)
dim = 768
heads = 8
x = torch.randn(2, 512, dim)

model = Transformer(
            hidden_dim=dim,
            num_head=heads,
            num_output=10,
            num_dec=6
        )

outputs = model(x)
print(outputs)
