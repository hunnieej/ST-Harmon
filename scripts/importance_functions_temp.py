import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from scripts.main_functions import mask_from_perm_keep_last
from typing import Optional, Tuple

@torch.inference_mode()
def cfg_delta_to_score(
    cond: torch.Tensor,
    uncd: torch.Tensor,
    *,
    score_mode: str = "l2",     # "l2"|"l1"|"linf"|"cos"|"rel_l2"|"rel_l1"|"dot"
    batch_reduce: str = "mean", # "mean"|"median"|"max"
    eps: float = 1e-6,
) -> torch.Tensor:
    if cond.shape != uncd.shape:
        raise ValueError(f"cond/uncd shape mismatch: {cond.shape} vs {uncd.shape}")
    if cond.dim() != 3:
        raise ValueError(f"expected (half,M,D), got {cond.shape}")

    cond_f = cond.float()
    uncd_f = uncd.float()
    delta = cond_f - uncd_f  # (half,M,D)

    if score_mode == "l2":
        # ||delta||_2
        per = torch.linalg.vector_norm(delta, ord=2, dim=-1)

    elif score_mode == "l1":
        # ||delta||_1
        per = delta.abs().sum(dim=-1)

    elif score_mode == "linf":
        # ||delta||_inf
        per = delta.abs().amax(dim=-1)

    elif score_mode == "cos":
        # cosine distance: 1 - cos(cond, uncd)
        # "방향 변화"를 보는 경향(크기 변화는 덜 민감)
        per = 1.0 - F.cosine_similarity(cond_f, uncd_f, dim=-1, eps=eps)

    elif score_mode == "rel_l2":
        # 상대 변화량: ||cond-uncd|| / (||uncd|| + eps)
        num = torch.linalg.vector_norm(delta, ord=2, dim=-1)
        den = torch.linalg.vector_norm(uncd_f, ord=2, dim=-1).clamp_min(eps)
        per = num / den

    elif score_mode == "rel_l1":
        num = delta.abs().sum(dim=-1)
        den = uncd_f.abs().sum(dim=-1).clamp_min(eps)
        per = num / den

    elif score_mode == "dot":
        # delta와 uncond의 정렬 정도(변형이 "기존 방향"을 따라가는지)
        # 해석이 까다로워서 보조 실험용 권장
        per = (delta * uncd_f).sum(dim=-1)
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
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
    global_tokens: torch.Tensor,
    global_mask: torch.Tensor,
    read_lin_list: List[torch.Tensor],
    writer_lin_list: List[torch.Tensor],
    writer_pix_list: List[torch.Tensor],
    tile_m: int,
    tile_n: int,
    orders_tile: torch.Tensor,
    temperature: float,
    device: torch.device,
    score_mode: str ="l2",
    batch_reduce: str ="mean",
) -> torch.Tensor:
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

# NOTE : ver.1 : Naive version : set global perm from importance map directly
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

# NOTE : ver.2 : fixed known cells, reorder unknown only
@torch.inference_mode()
def make_global_perm_fixed_known(
    *,
    score_map: torch.Tensor,          # (m,n) importance, high => earlier
    global_mask: torch.Tensor,        # (bsz,m,n) float 0/1 (1 unknown)
    prev_perm: Optional[torch.Tensor] = None,  # (bsz,L) for stable known ordering
    noise_std: float = 0.01,
 ) -> torch.Tensor:
    """
    Safe permutation builder for mid-run re-ordering.

     Invariant we MUST satisfy to avoid re-masking:
       - already known tokens (mask==0) must be placed in the PREFIX
       - unknown tokens (mask==1) must be placed in the SUFFIX
     Because mask_from_perm_keep_last(perm, k) keeps the LAST k indices as unknown.
     If known indices leak into the suffix, they can become unknown again.

     Strategy:
       perm = [ known_in_prev_order , unknown_sorted_by_score_desc ]

     Notes:
       - unknown is sorted by score desc (with small noise tie-break if requested)
       - known keeps the relative order from prev_perm when provided (for stability)
       - cond/uncond alignment: second half copies the first half (paired CFG batch)
    """
    if score_map.dim() != 2:
        raise ValueError("score_map must be (m,n)")
    if global_mask.dim() != 3:
        raise ValueError("global_mask must be (bsz,m,n)")

    bsz, m, n = global_mask.shape
    L = m * n
    device = global_mask.device

    score_flat = score_map.reshape(-1).float().to(device)  # (L,)
    mask_flat = global_mask.view(bsz, L).float()
    unknown_bool = mask_flat > 0.5
    known_bool = ~unknown_bool

    if prev_perm is None:
        prev_perm = torch.arange(L, device=device, dtype=torch.long).view(1, L).expand(bsz, L)
    else:
        if prev_perm.shape != (bsz, L):
            raise ValueError(f"prev_perm shape mismatch: {prev_perm.shape} vs {(bsz,L)}")
        prev_perm = prev_perm.to(device=device, dtype=torch.long)

    perm_out = torch.empty((bsz, L), device=device, dtype=torch.long)

    for b in range(bsz):
        pprev = prev_perm[b]  # (L,)

        # keep known in previous relative order (stable)
        known_sel = known_bool[b].index_select(0, pprev)  # bool over perm positions
        known_in_prev = pprev[known_sel]                  # (K_known,)

        # sort unknown by score desc (with optional noise tie-break)
        unknown_idx = torch.nonzero(unknown_bool[b], as_tuple=False).squeeze(1)  # (K_unknown,)
        if unknown_idx.numel() > 0:
            scores = score_flat.index_select(0, unknown_idx)
            if noise_std and noise_std > 0:
                scores = scores + noise_std * torch.randn_like(scores)
            order = torch.argsort(scores, descending=True)
            unknown_sorted = unknown_idx.index_select(0, order)
        else:
            unknown_sorted = unknown_idx

        perm_new = torch.cat([known_in_prev, unknown_sorted], dim=0)
        if perm_new.numel() != L:
            raise RuntimeError(f"[make_global_perm_fixed_known] perm size mismatch: {perm_new.numel()} vs {L}")
        perm_out[b] = perm_new

    # cond/uncond alignment (paired batch)
    if bsz % 2 == 0:
        half = bsz // 2
        perm_out = perm_out.clone()
        perm_out[half:] = perm_out[:half]

    return perm_out

@torch.inference_mode()
def _neighbors_from_indices(idx: torch.Tensor, m: int, n: int, mode: int = 4) -> torch.Tensor:
    """
    idx: (K,) linear indices in [0, m*n)
    returns: unique neighbor indices (including anchors themselves is OK; later we set unique)
    mode: "4" or "8"
    """
    r = idx // n
    c = idx % n

    # candidate offsets
    if mode == 4:
        dr = torch.tensor([-1, 1, 0, 0], device=idx.device)
        dc = torch.tensor([0, 0, -1, 1], device=idx.device)
    elif mode == 8:
        dr = torch.tensor([-1,-1,-1, 0,0, 1,1,1], device=idx.device)
        dc = torch.tensor([-1, 0, 1,-1,1,-1,0,1], device=idx.device)
    else:
        raise ValueError("mode must be 4 or 8")

    rr = r[:, None] + dr[None, :]
    cc = c[:, None] + dc[None, :]
    valid = (rr >= 0) & (rr < m) & (cc >= 0) & (cc < n)
    rr = rr[valid]
    cc = cc[valid]
    neigh = rr * n + cc
    return neigh

# NOTE : ver.3 : anchor + neighbors
@torch.inference_mode()
def make_perm_anchor_neighbors(
    *,
    score_map: torch.Tensor,           # (m,n) float, high => important
    global_mask: torch.Tensor,         # (bsz,m,n) float 0/1 (1 unknown)
    prev_perm: Optional[torch.Tensor] = None,  # (bsz,L)
    topk: int = 32,
    neighbor_mode: int = 4,          # "4" or "8"
    include_ring2: bool = False,       # optionally expand one more ring
    noise_std: float = 0.0,            # tie-break
) -> torch.Tensor:
    """
    perm = [known_prefix (stable) ; unknown_priority (anchors+neighbors) ; unknown_rest (by score desc)]
    Compatible with mask_from_perm_keep_last().
    """
    assert score_map.dim() == 2
    assert global_mask.dim() == 3
    bsz, m, n = global_mask.shape
    L = m * n
    device = global_mask.device

    score_flat = score_map.reshape(-1).float().to(device)      # (L,)
    mask_flat = global_mask.view(bsz, L).float()
    unknown = mask_flat > 0.5
    known = ~unknown

    if prev_perm is None:
        prev_perm = torch.arange(L, device=device).view(1, L).expand(bsz, L).clone()
    else:
        prev_perm = prev_perm.to(device=device, dtype=torch.long)

    perm_out = torch.empty((bsz, L), device=device, dtype=torch.long)

    for b in range(bsz):
        # stable known prefix (keep relative order from prev_perm)
        pprev = prev_perm[b]
        known_sel = known[b].index_select(0, pprev)
        known_prefix = pprev[known_sel]  # (K_known,)

        # candidate anchors among UNKNOWN only
        u_idx = torch.nonzero(unknown[b], as_tuple=False).squeeze(1)
        if u_idx.numel() == 0:
            perm_out[b] = pprev
            continue

        u_scores = score_flat.index_select(0, u_idx)
        if noise_std > 0:
            u_scores = u_scores + noise_std * torch.randn_like(u_scores)

        # pick top-k anchors (on unknown)
        k = min(topk, u_idx.numel())
        anchor_order = torch.argsort(u_scores, descending=True)[:k]
        anchors = u_idx.index_select(0, anchor_order)  # (k,)

        # build priority set = anchors + neighbors (restricted to unknown)
        neigh1 = _neighbors_from_indices(anchors, m, n, mode=neighbor_mode)
        priority = torch.cat([anchors, neigh1], dim=0)

        if include_ring2:
            neigh2 = _neighbors_from_indices(neigh1.unique(), m, n, mode=neighbor_mode)
            priority = torch.cat([priority, neigh2], dim=0)

        # unique + keep only unknown
        priority = priority.unique()
        priority = priority[unknown[b].index_select(0, priority)]

        # rest unknown = unknown - priority
        # make a boolean mask over L for priority
        pr_mask = torch.zeros(L, device=device, dtype=torch.bool)
        pr_mask[priority] = True
        rest_unknown = u_idx[~pr_mask.index_select(0, u_idx)]

        # sort priority and rest_unknown by score desc (optional but usually helpful)
        pr_scores = score_flat.index_select(0, priority)
        pr_sort = torch.argsort(pr_scores, descending=True)
        priority_sorted = priority.index_select(0, pr_sort)

        ru_scores = score_flat.index_select(0, rest_unknown)
        ru_sort = torch.argsort(ru_scores, descending=True)
        rest_sorted = rest_unknown.index_select(0, ru_sort)

        # unknown suffix = [priority_sorted, rest_sorted]
        unknown_suffix = torch.cat([priority_sorted, rest_sorted], dim=0)

        perm_new = torch.cat([known_prefix, unknown_suffix], dim=0)
        if perm_new.numel() != L:
            raise RuntimeError(f"perm size mismatch {perm_new.numel()} vs {L}")
        perm_out[b] = perm_new

    # paired CFG batch alignment
    if bsz % 2 == 0:
        half = bsz // 2
        perm_out[half:] = perm_out[:half]

    return perm_out

@torch.inference_mode()
def _neighbors_of_idx(idx: torch.Tensor, m: int, n: int, mode: int = 4) -> torch.Tensor:
    """
    idx: (K,) linear indices
    return: (K * deg,) neighbor linear indices (not unique), clipped to valid range
    """
    r = idx // n
    c = idx % n

    if mode == 4:
        dr = torch.tensor([-1, 1, 0, 0], device=idx.device)
        dc = torch.tensor([0, 0, -1, 1], device=idx.device)
    elif mode == 8:
        dr = torch.tensor([-1,-1,-1, 0,0, 1,1,1], device=idx.device)
        dc = torch.tensor([-1, 0, 1,-1,1,-1,0,1], device=idx.device)
    else:
        raise ValueError("mode must be '4' or '8'")

    rr = r[:, None] + dr[None, :]
    cc = c[:, None] + dc[None, :]
    valid = (rr >= 0) & (rr < m) & (cc >= 0) & (cc < n)

    rr = rr[valid]
    cc = cc[valid]
    return rr * n + cc

# NOTE : ver.4 : anchor then neighbors
@torch.inference_mode()
def make_perm_anchor_then_neighbors(
    *,
    score_map: torch.Tensor,                # (m,n) float, higher => more important
    global_mask: torch.Tensor,              # (bsz,m,n) float 0/1 (1 unknown, 0 known)
    topk: int = 32,                         # number of anchors
    neighbor_mode: str = "4",               # "4" or "8"
    prev_perm: Optional[torch.Tensor] = None,  # (bsz,L) for stable known ordering
    noise_std: float = 0.0,                 # tie-break noise for anchor ranking
    keep_anchor_neighbors_order: bool = True,  # True: anchor then its neighbors in any order
) -> torch.Tensor:
    """
    Build perm such that unknown order is:
      a1, N(a1), a2, N(a2), ..., ak, N(ak), (then rest unknown by score desc)

    Safe invariant (for mask_from_perm_keep_last):
      perm = [known_prefix ; unknown_suffix]

    Notes:
      - neighbors are added immediately after each anchor (your requested behavior)
      - all selected indices are restricted to UNKNOWN only
      - duplicates are removed; once a token is added, it won't be added again
      - remaining unknown tokens (not in anchors/neighbors) are appended by score desc
      - paired CFG batch alignment is enforced if bsz is even
    """
    if score_map.dim() != 2:
        raise ValueError("score_map must be (m,n)")
    if global_mask.dim() != 3:
        raise ValueError("global_mask must be (bsz,m,n)")

    bsz, m, n = global_mask.shape
    L = m * n
    device = global_mask.device

    score_flat = score_map.reshape(-1).float().to(device)    # (L,)
    mask_flat = global_mask.view(bsz, L).float()
    unknown = mask_flat > 0.5
    known = ~unknown

    if prev_perm is None:
        prev_perm = torch.arange(L, device=device, dtype=torch.long).view(1, L).expand(bsz, L)
    else:
        if prev_perm.shape != (bsz, L):
            raise ValueError(f"prev_perm shape mismatch: {prev_perm.shape} vs {(bsz,L)}")
        prev_perm = prev_perm.to(device=device, dtype=torch.long)

    perm_out = torch.empty((bsz, L), device=device, dtype=torch.long)

    for b in range(bsz):
        pprev = prev_perm[b]

        # ---- known prefix: keep known in prev_perm relative order (stable) ----
        known_sel = known[b].index_select(0, pprev)   # bool aligned to perm positions
        known_prefix = pprev[known_sel]               # (K_known,)

        # ---- anchors: pick top-k among UNKNOWN only ----
        u_idx = torch.nonzero(unknown[b], as_tuple=False).squeeze(1)  # (K_unknown,)
        if u_idx.numel() == 0:
            perm_out[b] = torch.cat([known_prefix, u_idx], dim=0)
            continue

        u_scores = score_flat.index_select(0, u_idx)
        if noise_std and noise_std > 0:
            u_scores = u_scores + noise_std * torch.randn_like(u_scores)

        k = min(topk, u_idx.numel())
        anchor_order = torch.argsort(u_scores, descending=True)[:k]
        anchors = u_idx.index_select(0, anchor_order)  # (k,)

        # ---- build unknown suffix: a1, N(a1), a2, N(a2), ... ----
        chosen = torch.zeros(L, device=device, dtype=torch.bool)  # track already added (within unknown suffix)
        suffix_list = []

        def try_add(idxs: torch.Tensor):
            """Add indices in order if they are unknown and not already chosen."""
            if idxs.numel() == 0:
                return
            # keep only unknown
            idxs = idxs[unknown[b].index_select(0, idxs)]
            if idxs.numel() == 0:
                return
            # remove already chosen
            new_mask = ~chosen.index_select(0, idxs)
            idxs = idxs[new_mask]
            if idxs.numel() == 0:
                return
            chosen[idxs] = True
            suffix_list.append(idxs)

        for a in anchors:
            a = a.view(1)
            try_add(a)

            neigh = _neighbors_of_idx(a, m, n, mode=neighbor_mode)  # (deg,)
            if keep_anchor_neighbors_order:
                # add neighbors in the natural order produced by offsets
                try_add(neigh)
            else:
                # optionally sort neighbors by score desc
                if neigh.numel() > 0:
                    neigh = neigh.unique()
                    neigh_scores = score_flat.index_select(0, neigh)
                    neigh = neigh.index_select(0, torch.argsort(neigh_scores, descending=True))
                try_add(neigh)

        # ---- append the rest unknown (not chosen) by score desc ----
        rest_unknown = u_idx[~chosen.index_select(0, u_idx)]
        if rest_unknown.numel() > 0:
            rest_scores = score_flat.index_select(0, rest_unknown)
            rest_sorted = rest_unknown.index_select(0, torch.argsort(rest_scores, descending=True))
        else:
            rest_sorted = rest_unknown

        if len(suffix_list) > 0:
            priority_suffix = torch.cat(suffix_list, dim=0)
        else:
            priority_suffix = torch.empty((0,), device=device, dtype=torch.long)

        unknown_suffix = torch.cat([priority_suffix, rest_sorted], dim=0)

        perm_new = torch.cat([known_prefix, unknown_suffix], dim=0)
        if perm_new.numel() != L:
            raise RuntimeError(f"perm size mismatch: {perm_new.numel()} vs {L}")
        perm_out[b] = perm_new

    # ---- paired CFG batch alignment ----
    if bsz % 2 == 0:
        half = bsz // 2
        perm_out = perm_out.clone()
        perm_out[half:] = perm_out[:half]

    return perm_out

def build_perm_by_phase(step, importance_map, global_mask, prev_perm, num_iter, anchor_topk):
    # phase boundaries
    s1 = int(num_iter * 0.25)
    s2 = int(num_iter * 0.50)

    # Phase 0: early
    if step < s1:
        # safest: reorder unknown only, no neighbor forcing
        return make_global_perm_fixed_known(
            score_map=importance_map,
            global_mask=global_mask,
            prev_perm=prev_perm,   # step0이면 None 가능, update면 prev_perm 사용
            noise_std=0.01,
        )

    # Phase 1: mid
    elif step < s2:
        return make_perm_anchor_neighbors(
            score_map=importance_map,
            global_mask=global_mask,
            prev_perm=prev_perm,   # IMPORTANT: keep stability
            topk=anchor_topk,
            neighbor_mode=4,       # weaker neighbor
            noise_std=0.01,
        )

    # Phase 2: late
    else:
        # go back to conservative (or keep anchor_then w/ smaller topk)
        return make_global_perm_fixed_known(
            score_map=importance_map,
            global_mask=global_mask,
            prev_perm=prev_perm,
            noise_std=0.0,         # late stage: reduce churn
        )

# NOTE : ver.5 : Solving top-k redundancy
@torch.inference_mode()
def _select_anchors_stratified_random(
    unknown_bool_2d: torch.Tensor,  # (m,n) bool
    K: int,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Region-wise stratified sampling: split grid into bins, sample 1 unknown per bin.
    Falls back to global random if bins under-fill.
    returns: (k_sel,) long of linear indices (<=K)
    """
    device = unknown_bool_2d.device
    m, n = unknown_bool_2d.shape
    if K <= 0:
        return torch.empty(0, device=device, dtype=torch.long)

    # Choose bin counts respecting aspect ratio
    r_bins = max(1, int(round(math.sqrt(K * m / max(n, 1)))))
    c_bins = max(1, int(math.ceil(K / r_bins)))

    r_edges = torch.linspace(0, m, steps=r_bins + 1, device=device).long()
    c_edges = torch.linspace(0, n, steps=c_bins + 1, device=device).long()

    picks = []
    for ri in range(r_bins):
        r0, r1 = int(r_edges[ri].item()), int(r_edges[ri + 1].item())
        for ci in range(c_bins):
            if len(picks) >= K:
                break
            c0, c1 = int(c_edges[ci].item()), int(c_edges[ci + 1].item())
            cell = unknown_bool_2d[r0:r1, c0:c1]
            if not cell.any():
                continue
            coords = torch.nonzero(cell, as_tuple=False)  # (Nc,2) in cell coords
            j = torch.randint(0, coords.shape[0], (1,), device=device, generator=generator).item()
            rr = coords[j, 0].item() + r0
            cc = coords[j, 1].item() + c0
            picks.append(rr * n + cc)

    if len(picks) < K:
        # fill remaining from all unknown randomly (without duplicates)
        all_unknown = torch.nonzero(unknown_bool_2d.view(-1), as_tuple=False).squeeze(1)
        if all_unknown.numel() > 0:
            # remove already picked
            if len(picks) > 0:
                picked = torch.tensor(picks, device=device, dtype=torch.long)
                mask = torch.ones(all_unknown.numel(), device=device, dtype=torch.bool)
                # O(U*K) but K small; ok
                for p in picked:
                    mask &= (all_unknown != p)
                remain = all_unknown[mask]
            else:
                remain = all_unknown
            if remain.numel() > 0:
                need = K - len(picks)
                perm = torch.randperm(remain.numel(), device=device, generator=generator)
                add = remain.index_select(0, perm[: min(need, remain.numel())])
                picks.extend(add.tolist())

    # Dedup (just in case) + trim
    if len(picks) == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    picks = list(dict.fromkeys(picks))  # preserve order
    picks = picks[:K]
    return torch.tensor(picks, device=device, dtype=torch.long)

@torch.inference_mode()
def _select_diverse_anchors_from_pool(
    score_flat: torch.Tensor,       # (L,) float
    unknown_idx: torch.Tensor,      # (U,) long
    m: int,
    n: int,
    K: int,
    *,
    pool_mult: int = 4,
    alpha: float = 0.7,
    dist_mode: str = "manhattan",   # "manhattan" | "euclidean"
    noise_std: float = 0.0,
) -> torch.Tensor:
    """
    Build pool from top-P scores among unknown, then pick K anchors with greedy diversity:
      a1 = argmax(score)
      aj = argmax score^alpha * min_dist_to_selected
    returns: (k_sel,) long anchor indices (linear)
    """
    device = unknown_idx.device
    U = unknown_idx.numel()
    if K <= 0 or U == 0:
        return torch.empty(0, device=device, dtype=torch.long)

    K_eff = min(K, U)
    P = min(U, max(K_eff, pool_mult * K_eff))

    scores_u = score_flat.index_select(0, unknown_idx).float()  # (U,)
    if noise_std and noise_std > 0:
        scores_u = scores_u + noise_std * torch.randn_like(scores_u)

    # pool = top-P among unknown
    topv, topi = torch.topk(scores_u, k=P, largest=True, sorted=False)
    pool_idx = unknown_idx.index_select(0, topi)  # (P,)
    pool_scores = topv                            # (P,)

    # coords for pool
    pr = (pool_idx // n).float()
    pc = (pool_idx % n).float()

    # select a1
    a_list = []
    a0 = torch.argmax(pool_scores).item()
    a_list.append(a0)

    # greedy pick next anchors
    selected = torch.zeros(P, device=device, dtype=torch.bool)
    selected[a0] = True

    # precompute score^alpha (clamp to avoid nan if negative)
    s_pow = torch.clamp(pool_scores, min=0.0).pow(alpha)

    for _ in range(1, K_eff):
        sel_idx = torch.nonzero(selected, as_tuple=False).squeeze(1)
        ar = pr.index_select(0, sel_idx)  # (S,)
        ac = pc.index_select(0, sel_idx)

        # distances: (P,S)
        if dist_mode == "euclidean":
            dr = pr[:, None] - ar[None, :]
            dc = pc[:, None] - ac[None, :]
            dist = torch.sqrt(dr * dr + dc * dc + 1e-6)
        else:
            dist = (pr[:, None] - ar[None, :]).abs() + (pc[:, None] - ac[None, :]).abs()

        min_dist = dist.min(dim=1).values  # (P,)
        obj = s_pow * (min_dist + 1.0)     # +1 to keep nonzero
        obj[selected] = -1e18              # already selected

        j = torch.argmax(obj).item()
        selected[j] = True
        a_list.append(j)

    anchors = pool_idx.index_select(0, torch.tensor(a_list, device=device, dtype=torch.long))
    return anchors  # (K_eff,)

@torch.inference_mode()
def make_global_perm_pool_diverse_anchors(
    *,
    score_map: torch.Tensor,                 # (m,n)
    global_mask: torch.Tensor,               # (bsz,m,n) float (1 unknown, 0 known)
    prev_perm: Optional[torch.Tensor] = None,# (bsz,L) for stable known ordering
    topk: int = 32,                          # K anchors
    pool_mult: int = 4,                      # P = pool_mult*K
    alpha: float = 0.7,
    dist_mode: str = "manhattan",
    noise_std: float = 0.01,                 # tie-break inside score
    # warmup: early spatial stratified anchors
    step: Optional[int] = None,
    warmup_steps: int = 0,                   # e.g., int(num_iter*0.15)
) -> torch.Tensor:
    """
    Perm invariant:
      perm = [ known_prefix_in_prev_order , anchors_first , remaining_unknown_by_score ]
    anchors are chosen:
      - if step < warmup_steps: stratified spatial random
      - else: pool-diverse greedy from top-P
    """
    if score_map.dim() != 2:
        raise ValueError("score_map must be (m,n)")
    if global_mask.dim() != 3:
        raise ValueError("global_mask must be (bsz,m,n)")

    bsz, m, n = global_mask.shape
    L = m * n
    device = global_mask.device

    score_flat = score_map.reshape(-1).float().to(device)  # (L,)
    mask_flat = global_mask.view(bsz, L).float()
    unknown_bool = mask_flat > 0.5
    known_bool = ~unknown_bool

    if prev_perm is None:
        prev_perm = torch.arange(L, device=device, dtype=torch.long).view(1, L).expand(bsz, L)
    else:
        if prev_perm.shape != (bsz, L):
            raise ValueError(f"prev_perm shape mismatch: {prev_perm.shape} vs {(bsz,L)}")
        prev_perm = prev_perm.to(device=device, dtype=torch.long)

    perm_out = torch.empty((bsz, L), device=device, dtype=torch.long)

    for b in range(bsz):
        pprev = prev_perm[b]

        # known prefix: keep stable order from prev_perm
        known_sel = known_bool[b].index_select(0, pprev)
        known_in_prev = pprev[known_sel]

        # unknown set
        unknown_idx = torch.nonzero(unknown_bool[b], as_tuple=False).squeeze(1)
        if unknown_idx.numel() == 0:
            perm_out[b] = torch.cat([known_in_prev, unknown_idx], dim=0)
            continue

        # choose anchors
        if (step is not None) and (step < warmup_steps):
            # stratified spatial random over current unknown mask
            unk2d = unknown_bool[b].view(m, n)
            anchors = _select_anchors_stratified_random(unk2d, K=topk)
        else:
            anchors = _select_diverse_anchors_from_pool(
                score_flat, unknown_idx, m, n, K=topk,
                pool_mult=pool_mult, alpha=alpha, dist_mode=dist_mode, noise_std=noise_std
            )

        # remove anchors from unknown for "remaining_unknown_by_score"
        if anchors.numel() > 0:
            # build a boolean keep mask for unknown_idx
            keep = torch.ones(unknown_idx.numel(), device=device, dtype=torch.bool)
            for a in anchors:
                keep &= (unknown_idx != a)
            remain = unknown_idx[keep]
        else:
            remain = unknown_idx

        # sort remaining by score desc
        if remain.numel() > 0:
            s = score_flat.index_select(0, remain)
            if noise_std and noise_std > 0:
                s = s + noise_std * torch.randn_like(s)
            order = torch.argsort(s, descending=True)
            remain_sorted = remain.index_select(0, order)
        else:
            remain_sorted = remain

        unknown_sorted = torch.cat([anchors, remain_sorted], dim=0)
        perm_new = torch.cat([known_in_prev, unknown_sorted], dim=0)

        if perm_new.numel() != L:
            raise RuntimeError(f"perm size mismatch: {perm_new.numel()} vs {L}")
        perm_out[b] = perm_new

    # cond/uncond alignment (paired CFG batch)
    if bsz % 2 == 0:
        half = bsz // 2
        perm_out = perm_out.clone()
        perm_out[half:] = perm_out[:half]

    return perm_out

#---- Visualization functions ----

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
    rank2d = 1.0 - rank2d
    arr = rank2d.numpy()

    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=_make_figsize(m, n, base_h=base_h))
    plt.imshow(arr, origin="upper", interpolation="nearest", aspect="equal")
    plt.colorbar()
    plt.title("Rank map")
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
