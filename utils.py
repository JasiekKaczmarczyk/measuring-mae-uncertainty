import torch 
from rollout import rollout

def attention_head_fusion(attention_map: torch.Tensor, fusion_type: str = "mean"):
    # shape [batch_size, num_heads, seq_len, seq_len]
    if fusion_type == "mean":
        return attention_map.mean(dim=1)
    elif fusion_type == "min":
        return attention_map.amin(dim=1)
    elif fusion_type == "max":
        return attention_map.amax(dim=1)
    else:
        raise NotImplementedError()
    
def aggregate_attention_maps(attentions: list[torch.Tensor], aggregation_type: str = "last", fusion_type: str = "mean"):
    # shape [batch_size, num_heads, seq_len, seq_len] x num_attention_blocks
    if aggregation_type == "last":
        attn = attentions[-1].detach()
        attn = attention_head_fusion(attn, fusion_type=fusion_type)
        attn = remove_cls_token(attn)
    elif aggregation_type == "mean":
        attn = torch.zeros_like(attentions[0])
        for attention in attentions:
            attn += attention
        attn = attn.detach() / len(attentions)
        attn = attention_head_fusion(attn, fusion_type=fusion_type)
        attn = remove_cls_token(attn)
    elif aggregation_type == "rollout":
        attn = rollout(attentions, head_fusion=fusion_type).detach()
        attn = remove_cls_token(attn)
    else:
        raise NotImplementedError()
    
    # returns shape: [batch_size, seq_len, seq_len]
    return attn

def remove_cls_token(attention: torch.Tensor):
    # shape: [batch_size, num_heads, cls_token+seq_len, cls_token+seq_len] -> [batch_size, num_heads, seq_len, seq_len]
    return attention[:, 1:, 1:]