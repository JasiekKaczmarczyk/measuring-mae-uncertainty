import math
import torch
import einops

def mae_loss(pred: torch.Tensor, target: torch.Tensor, use_norm_pix_loss: bool = False):
    if use_norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    return loss

def shannon_entropy(attn: torch.Tensor):
    # attn shape: [batch_size, seq_len, seq_len]
    # base is seq_len
    base = attn.shape[-1]
    # shape: [batch_size, seq_len]
    H = -torch.sum(attn * torch.log(attn), dim=-1) / math.log(base)

    return H

def gini_index(attn: torch.Tensor):
    # attn shape: [batch_size, seq_len, seq_len]
    G = 1 - torch.sum(attn ** 2, dim=-1)

    return G

def batch_cov(attn: torch.Tensor):
    # shape: [batch_size, query_len, key_len]
    B, H, Q, K = attn.shape
    attn = einops.rearrange(attn, "b h q k -> b q h k")

    # shape: [B. Q, H, 1]
    mean = attn.mean(dim=-1, keepdim=True)
    # shape: [B, Q, H, K]
    diffs = attn - mean

    prods = torch.einsum("b q h k, b q j k -> b q h j", [diffs, diffs])
    bcov = prods / (K - 1)  # Unbiased estimate

    return bcov

def attention_spread(attn: torch.Tensor):
    # max_abs_value = torch.abs(torch.amax(attn, dim=-1, keepdim=True))
    # attn = attn / max_abs_value
    # attn = torch.log(attn)
    attn = (attn - torch.mean(attn, dim=-1, keepdim=True)) / torch.std(attn, dim=-1, keepdim=True)

    # shapes: [batch_size, num_heads, seq_len, seq_len] -> [batch_size, seq_len, num_heads, num_heads]
    bcov = batch_cov(attn)

    attn_spread = torch.abs(torch.linalg.det(bcov))

    return attn_spread

def attention_focus_score(attn: torch.Tensor):
    
    return 1/torch.sum(attn**2, dim=-1)

def KL_divergence(attn: torch.Tensor):
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    uniform = torch.nn.functional.log_softmax(torch.randn(attn.shape, requires_grad=True), dim=-1)
    attn = torch.nn.functional.log_softmax(attn, dim=1)

    kl_divs = torch.zeros(attn.shape[0], attn.shape[1])

    for i in range(attn.shape[0]):
        for j in range(attn.shape[1]):
            kl_div = kl_loss(attn[i, j, :], uniform[i, j, :])
            kl_divs[i, j] = kl_div.mean()

    return kl_divs

def maximum_attention_weight(attn: torch.Tensor):
    return torch.max(attn, dim=-1)[0]

