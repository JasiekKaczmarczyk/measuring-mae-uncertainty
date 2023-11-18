import torch
import einops

# chyba jednak niepotrzebne bo można to zrobić za pomocą model.patchify() lub model.unpatchify(), ale zostawię na wszelki wypadek

def patchify(imgs: torch.Tensor, patch_size: int = 16):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

    # h, w = imgs.shape[2] // patch_size, imgs.shape[3] // patch_size
    x = einops.rearrange(imgs, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)
    return x

def unpatchify(x: torch.Tensor, patch_size: int = 16):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    imgs = einops.rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, p1=patch_size, p2=patch_size)

    return imgs