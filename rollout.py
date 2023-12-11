import torch
import numpy as np

def rollout(attentions: list[torch.Tensor], head_fusion: str = "mean"):
    num_patches = attentions[0].size(-1)

    # shape: [num_patches, num_patches]
    result = torch.eye(num_patches)
    with torch.no_grad():
        for attention in attentions:
            # shape: [batch_size, num_patches, num_patches]
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.amax(axis=1)
            elif head_fusion == "min":
                attention_heads_fused = attention.amin(axis=1)
            else:
                raise "Attention head fusion type Not supported"

            I = torch.eye(num_patches)
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = a @ result
    
    return result