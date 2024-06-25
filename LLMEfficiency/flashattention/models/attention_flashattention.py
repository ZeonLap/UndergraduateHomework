import torch
import torch.nn as nn
import math
from flash_attn import flash_attn_func


class FlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.dropout = config["attention_dropout"]
        self.head_dim = config["head_dim"]

    def forward(self, q, k, v, attn_mask=None):
        q = (q * attn_mask[:, None, :, None]).half()
        k = (k * attn_mask[:, None, :, None]).half()
        v = (v * attn_mask[:, None, :, None]).half()

        scale = q.size(-1) ** -0.5
        return flash_attn_func(q, k, v, self.dropout, scale)
