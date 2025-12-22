from pathlib import Path
import torch
from PIL import Image

def save_tensor_image(x: torch.Tensor, path: Path):
    """
    x: (3, H, W) float tensor in [-1, 1]
    """
    x = torch.clamp(127.5 * x + 128.0, 0, 255).to("cpu", dtype=torch.uint8)
    x = x.permute(1, 2, 0).contiguous().numpy()  # HWC uint8
    Image.fromarray(x).save(str(path))

def expand_cfg_batch(input_ids_2, attn_2, base_bsz: int, cfg: float):
    """
    input_ids_2, attn_2: (2, T) = [cond, uncond] from prepare_text_conditions
    returns:
      - cfg!=1: (2*base_bsz, T) = [cond x base_bsz; uncond x base_bsz]
      - cfg==1: (base_bsz, T)   = [cond x base_bsz]
    """
    if cfg == 1.0:
        input_ids = input_ids_2[:1].expand(base_bsz, -1)
        attn = attn_2[:1].expand(base_bsz, -1)
        return input_ids, attn

    input_ids = torch.cat([
        input_ids_2[:1].expand(base_bsz, -1),
        input_ids_2[1:].expand(base_bsz, -1),
    ])
    attn = torch.cat([
        attn_2[:1].expand(base_bsz, -1),
        attn_2[1:].expand(base_bsz, -1),
    ])
    return input_ids, attn

def make_global_perm(bsz: int, L: int, device: torch.device) -> torch.Tensor:
    # per-sample permutation indices in prediction order (early->late)
    return torch.argsort(torch.rand(bsz, L, device=device), dim=-1)

def mask_from_perm_keep_last(perm: torch.Tensor, mask_len: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Build a mask (1=unknown, 0=known) that keeps the LAST `mask_len` indices in `perm` as unknown.
    perm: (B, L)
    """
    bsz, L = perm.shape
    mask = torch.zeros(bsz, L, device=perm.device, dtype=dtype)
    if mask_len <= 0:
        return mask
    idx = perm[:, -mask_len:]  # (B, mask_len)
    mask.scatter_(1, idx, 1.0)
    return mask