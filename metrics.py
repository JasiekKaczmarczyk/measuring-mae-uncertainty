import math
import torch

def shannon_entropy(attn: torch.Tensor):
    # attn shape: [batch_size, num_heads, seq_len, seq_len]
    # base is num_heads
    base = attn.shape[1]
    H = -torch.sum(attn * torch.log(attn), dim=-1) / math.log(base)

    return H

def attention_spread(attentions: list[torch.Tensor]):
    pass