import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from scripts.main_functions import mask_from_perm_keep_last

@torch.inference_mode()
def cfg_delta_to_score(
    cond: torch.Tensor,     # (half, M, D)
    uncd: torch.Tensor,     # (half, M, D)
    *,
    score_mode: str = "l2",     # "l2"|"l1"|"linf"|"cos"|"rel_l2"|"rel_l1"|"dot"
    batch_reduce: str = "mean", # "mean"|"median"|"max"
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Return:
      score: (M,) float32
      - larger => stronger text effect (cond vs uncond difference)
    """
    if cond.shape != uncd.shape:
        raise ValueError(f"cond/uncd shape mismatch: {cond.shape} vs {uncd.shape}")
    if cond.dim() != 3:
        raise ValueError(f"expected (half,M,D), got {cond.shape}")

    # float32로 안정화
    cond_f = cond.float()
    uncd_f = uncd.float()
    delta = cond_f - uncd_f  # (half,M,D)

    if score_mode == "l2":
        # ||delta||_2
        per = torch.linalg.vector_norm(delta, ord=2, dim=-1)  # (half,M)

    elif score_mode == "l1":
        # ||delta||_1
        per = delta.abs().sum(dim=-1)  # (half,M)

    elif score_mode == "linf":
        # ||delta||_inf
        per = delta.abs().amax(dim=-1)  # (half,M)

    elif score_mode == "cos":
        # cosine distance: 1 - cos(cond, uncd)
        # "방향 변화"를 보는 경향(크기 변화는 덜 민감)
        per = 1.0 - F.cosine_similarity(cond_f, uncd_f, dim=-1, eps=eps)  # (half,M)

    elif score_mode == "rel_l2":
        # 상대 변화량: ||cond-uncd|| / (||uncd|| + eps)
        num = torch.linalg.vector_norm(delta, ord=2, dim=-1)  # (half,M)
        den = torch.linalg.vector_norm(uncd_f, ord=2, dim=-1).clamp_min(eps)  # (half,M)
        per = num / den

    elif score_mode == "rel_l1":
        num = delta.abs().sum(dim=-1)  # (half,M)
        den = uncd_f.abs().sum(dim=-1).clamp_min(eps)  # (half,M)
        per = num / den

    elif score_mode == "dot":
        # delta와 uncond의 정렬 정도(변형이 "기존 방향"을 따라가는지)
        # 해석이 까다로워서 보조 실험용 권장
        per = (delta * uncd_f).sum(dim=-1)  # (half,M)
        per = per.abs()

    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    # batch reduce: (half,M) -> (M,)
    if batch_reduce == "mean":
        score = per.mean(dim=0)
    elif batch_reduce == "median":
        score = per.median(dim=0).values
    elif batch_reduce == "max":
        score = per.max(dim=0).values
    else:
        raise ValueError(f"Unknown batch_reduce: {batch_reduce}")

    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return score.float()  # (M,)

@torch.inference_mode()
def compute_global_cfg_importance_map(
    *,
    model,
    step: int,
    num_iter: int,
    input_ids: torch.Tensor,          # (bsz, T) = [cond...; uncond...]
    attention_mask: torch.Tensor,     # (bsz, T)
    past_key_values,
    global_tokens: torch.Tensor,      # (bsz, m, n, D)
    global_mask: torch.Tensor,        # (bsz, m, n) float 0/1 (1 unknown)
    read_lin_list: List[torch.Tensor],
    writer_lin_list: List[torch.Tensor],
    writer_pix_list: List[torch.Tensor],
    tile_m: int,
    tile_n: int,
    orders_tile: torch.Tensor,        # (bsz, M)
    temperature: float,
    device: torch.device,
    score_mode: str ="l2",
    batch_reduce: str ="mean",
) -> torch.Tensor:
    """
    CFG-delta 기반 global importance map 생성.
    출력:
      importance_map: (m, n) float32, [0,1]
      - 값이 클수록 cond/uncond 차이가 큰 위치(=텍스트 조건 영향이 큰 위치)
    """
    bsz, m, n, D = global_tokens.shape
    L = m * n
    M = tile_m * tile_n

    global_tokens_flat = global_tokens.view(bsz, L, D)
    global_mask_flat = global_mask.view(bsz, L)  # 1 unknown

    imp_flat = torch.zeros(L, device=device, dtype=torch.float32)

    if bsz % 2 != 0:
        raise ValueError("CFG-delta importance requires bsz to be even (cond/uncond paired).")
    half = bsz // 2

    for t in range(len(read_lin_list)):
        lin_read  = read_lin_list[t]     # (M,)
        lin_write = writer_lin_list[t]   # (K,)
        pix_write = writer_pix_list[t]   # (K,)

        tokens_in_flat = global_tokens_flat.index_select(1, lin_read)  # (bsz, M, D)
        mask_in_flat   = global_mask_flat.index_select(1, lin_read)    # (bsz, M)

        tokens_in = tokens_in_flat.view(bsz, tile_m, tile_n, D)
        mask_in   = mask_in_flat.view(bsz, tile_m, tile_n)

        # probe: 현재 unknown 픽셀만 예측하도록 설정 (불필요한 계산 줄임)
        mask_to_pred = mask_in.bool()
        mask_next = mask_in  # 상태 변경하지 않음 (score만 계산)

        # 매우 중요: cfg=1.0로 돌려서 cond/uncond를 섞지 않고 "각각"의 예측을 받는다
        out = model.sample_step_tokens(
            step=step,
            num_iter=num_iter,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokens=tokens_in,
            mask=mask_in,
            orders=orders_tile,
            past_key_values=past_key_values,
            cfg=1.0,
            cfg_schedule="constant",
            temperature=temperature,
            image_shape=(tile_m, tile_n),
            mask_to_pred=mask_to_pred,
            mask_next=mask_next,
        )

        toks_out = out["tokens"].view(bsz, M, D)
        score = cfg_delta_to_score(
            cond=toks_out[:half],
            uncd=toks_out[half:],
            score_mode=score_mode,
            batch_reduce=batch_reduce,
            eps=1e-6,
        )
        score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        # writer-pixel -> global cell
        score_writer = score.index_select(0, pix_write)
        imp_flat.index_copy_(0, lin_write, score_writer)

    imp = imp_flat.view(m, n)
    imp = imp - imp.min()
    imp = imp / imp.max().clamp_min(1e-6)
    return imp


@torch.inference_mode()
def make_global_perm_from_score_masked(
    *,
    score_map: torch.Tensor,     # (m,n)
    global_mask: torch.Tensor,   # (bsz,m,n) float 0/1 (1 unknown)
    noise_std: float = 0.01,
) -> torch.Tensor:
    """
    score high => earlier
    known(mask==0) => forced to the end
    """
    if score_map.dim() != 2:
        raise ValueError("score_map must be (m,n)")
    bsz, m, n = global_mask.shape
    L = m * n

    score = score_map.view(1, L).expand(bsz, L).contiguous().float()
    unknown = global_mask.view(bsz, L).float()

    # known은 뒤로: huge negative penalty
    score = score + (unknown - 1.0) * 1e9

    if noise_std > 0:
        score = score + noise_std * torch.randn_like(score)

    perm = torch.argsort(score, dim=-1, descending=True)

    # cond/uncond alignment
    if bsz % 2 == 0:
        half = bsz // 2
        perm = perm.clone()
        perm[half:] = perm[:half]
    return perm


def _make_figsize(m: int, n: int, base_h: float = 6.0):
    """
    Make figure size so that:
      - token cells are square (aspect='equal')
      - overall canvas matches n/m ratio
    """
    ratio = float(n) / float(m)  # width/height
    return (base_h * ratio, base_h)

def save_importance_heatmap(
    importance_map: torch.Tensor,
    out_path: Path,
    title: str = "Importance (token grid)",
    normalize: str = "minmax",  # "minmax" | "none"
    base_h: float = 6.0,
):
    """
    importance_map:
      - (m, n) or (bsz, m, n) torch tensor
    out_path:
      - e.g., out_dir / "importance_heatmap.png"
    """
    if importance_map.dim() == 3:
        imp = importance_map[0]
    else:
        imp = importance_map

    imp = imp.detach().float().cpu()

    if normalize == "minmax":
        mn = float(torch.min(imp).item())
        mx = float(torch.max(imp).item())
        imp = (imp - mn) / (mx - mn + 1e-8)

    arr = imp.numpy()
    m, n = arr.shape

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=_make_figsize(m, n, base_h=base_h))
    plt.imshow(arr, origin="upper", interpolation="nearest", aspect="equal")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close()

    npy_path = out_path.with_suffix(".npy")
    np.save(str(npy_path), arr)

    # print(f"[OK] Saved importance heatmap: {out_path}")
    # print(f"[OK] Saved importance array:   {npy_path}")


def save_rank_map(global_perm: torch.Tensor, m: int, n: int, out_png: Path, base_h: float = 6.0):
    """
    global_perm: (B, L) where L=m*n
    rank[x] = position in perm (0..L-1). smaller = earlier.
    """
    perm = global_perm[0].detach().cpu()  # (L,)
    L = perm.numel()
    rank = torch.empty(L, dtype=torch.long)
    rank[perm] = torch.arange(L, dtype=torch.long)
    rank2d = rank.view(m, n).float()

    rank2d = rank2d / (L - 1 + 1e-8)
    arr = rank2d.numpy()

    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=_make_figsize(m, n, base_h=base_h))
    plt.imshow(arr, origin="upper", interpolation="nearest", aspect="equal")
    plt.colorbar()
    plt.title("Rank map (earlier = darker)")
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=200)
    plt.close()

    print(f"[OK] Saved rank map: {out_png}")

def save_step_pred_mask(mask_to_pred_flat: torch.Tensor, m: int, n: int, out_png: Path, base_h: float = 6.0):
    """
    mask_to_pred_flat: (B, L) bool
    """
    mm = mask_to_pred_flat[0].view(m, n).detach().float().cpu()
    arr = mm.numpy()

    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=_make_figsize(m, n, base_h=base_h))
    plt.imshow(arr, origin="upper", interpolation="nearest", aspect="equal")
    plt.colorbar()
    plt.title("Predicted cells at this step")
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=200)
    plt.close()

def step_importance_curve(global_perm, importance_map, num_iter, device):
    # importance (m,n) -> (L,)
    imp = importance_map[0] if importance_map.dim()==3 else importance_map
    imp = imp.detach().float().to(device).view(-1)  # (L,)

    perm = global_perm[0].detach().to(device)  # (L,)
    L = perm.numel()

    # simulate your masking schedule exactly (bsz=1)
    mask = torch.ones(L, device=device, dtype=torch.float32)  # 1 unknown
    perm1 = perm.view(1, L)

    means = []
    for step in range(num_iter):
        if step >= num_iter - 1:
            to_pred = mask.bool()
            next_mask = torch.zeros_like(mask)
        else:
            mask_ratio = math.cos(math.pi / 2.0 * (step + 1) / num_iter)
            target_len = int(math.floor(L * mask_ratio))
            unknown0 = int(mask.sum().item())
            mask_len0 = max(1, min(unknown0 - 1, target_len))
            next_mask = mask_from_perm_keep_last(perm1, mask_len0, dtype=torch.float32)[0]
            to_pred = (mask.bool() ^ next_mask.bool())

        if to_pred.any():
            means.append(float(imp[to_pred].mean().item()))
        else:
            means.append(float("nan"))
        mask = next_mask

    return means
