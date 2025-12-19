'''
Docstring for scripts.text2pano_v3
- y-axis tiling 추가
'''

import argparse
import math
from pathlib import Path

import torch
from mmengine.config import Config
from PIL import Image
from tqdm import tqdm

from src.builder import BUILDER

import yaml
import re


def split_prompt_lmr(prompt: str):
    """
    Return (left, middle, right) clauses from a single sentence prompt.
    Expected pattern (naive):
      "... on the left, ... in the center, ... on the right."
    """
    p = prompt.strip()
    m = re.search(
        r"(.*?)(?:\s+on the left\s*,\s*)(.*?)(?:\s+in the center\s*,\s*)(.*?)(?:\s+on the right\.?\s*)$",
        p,
        flags=re.IGNORECASE
    )
    if m:
        left = m.group(1).strip()
        mid = m.group(2).strip()
        right = m.group(3).strip()
        return left, mid, right

    parts = [x.strip() for x in p.split(",") if x.strip()]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]

    raise ValueError("Prompt could not be split into [left]-[middle]-[right]. Please standardize the format.")


def save_tensor_image(x: torch.Tensor, path: Path):
    """
    x: (3, H, W) float tensor in [-1, 1]
    """
    x = torch.clamp(127.5 * x + 128.0, 0, 255).to("cpu", dtype=torch.uint8)
    x = x.permute(1, 2, 0).contiguous().numpy()  # HWC uint8
    Image.fromarray(x).save(str(path))


def make_global_perm(bsz: int, L: int, device: torch.device) -> torch.Tensor:
    """
    Returns a per-sample permutation of [0..L-1] with shape (bsz, L),
    interpreted as the ORDER OF PREDICTION (early -> late).
    """
    return torch.argsort(torch.rand(bsz, L, device=device), dim=-1)


def mask_from_perm_keep_last(perm: torch.Tensor, mask_len: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Build a mask (1=unknown, 0=known) that keeps the LAST `mask_len` indices in `perm` as unknown.
    perm: (B, L) permutation indices
    """
    bsz, L = perm.shape
    mask = torch.zeros(bsz, L, device=perm.device, dtype=dtype)
    if mask_len <= 0:
        return mask
    idx = perm[:, -mask_len:]  # (B, mask_len)
    mask.scatter_(1, idx, 1.0)
    return mask


def expand_cfg_batch(input_ids_2, attn_2, base_bsz: int, cfg: float):
    """
    input_ids_2, attn_2 are (2, T) = [cond, uncond] as returned by prepare_text_conditions.
    Returns expanded tensors with shape:
      - if cfg != 1.0: (2*base_bsz, T) = [cond x base_bsz; uncond x base_bsz]
      - else: (base_bsz, T)           = [cond x base_bsz]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file path.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a dog on the left, a cat in the center, a bird on the right.")
    parser.add_argument("--cfg_prompt", type=str, default="Generate an image.")
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg_schedule", type=str, default="constant")
    parser.add_argument("--num_iter", type=int, default=64)
    parser.add_argument("--grid_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=512)

    # IMPORTANT: keep these ratios as the global canvas size controller
    parser.add_argument("--height_ratio", type=int, default=2)   # e.g., 2 => 1024px height when image_size=512
    parser.add_argument("--width_ratio", type=int, default=4)    # e.g., 4 => 2048px width when image_size=512

    parser.add_argument("--output", type=str, default="output.jpg")
    parser.add_argument("--num_tiles", type=int, default=5)      # tiles along width
    parser.add_argument("--num_rows", type=int, default=3)       # tiles along height
    parser.add_argument("--pad_tokens", type=int, default=16)    # padding context in latent tokens

    parser.add_argument("--clause_repeat", type=int, default=2,
                        help="How many extra times to repeat the selected clause for region weighting (naive).")
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    # ----------------- log.yaml -----------------
    out_path = Path(args.output).resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "log.yaml"
    with open(log_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            vars(args),
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
    print(f"[OK] Saved log: {log_path}", flush=True)

    # ----------------- build model -----------------
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).eval().cuda()
    model = model.to(model.dtype)

    checkpoint = torch.load(args.checkpoint, map_location="cuda")
    _ = model.load_state_dict(checkpoint, strict=False)

    # ----------------- prompts: parse L/M/R -----------------
    total_prompt = args.prompt
    left_clause, mid_clause, right_clause = split_prompt_lmr(total_prompt)

    base_cond_prompt = f"Generate an image: {total_prompt}"
    print(base_cond_prompt, flush=True)

    base_bsz = args.grid_size ** 2

    # ----------------- global latent grid shape (KEEP ratios) -----------------
    # CHANGED: m,n now define the full canvas (height_ratio x width_ratio)
    m = (args.image_size * args.height_ratio) // 16
    n = (args.image_size * args.width_ratio) // 16
    L = m * n

    # ----------------- tile shape (FIXED to 512x512 per tile) -----------------
    # CHANGED: tile_m is NOT equal to m anymore; we tile in y as well.
    tile_m = args.image_size // 16
    tile_n = args.image_size // 16

    # ----------------- tiling along width (x) -----------------
    num_tiles = args.num_tiles
    if num_tiles < 2:
        raise ValueError("--num_tiles must be >= 2 for panorama tiling.")

    stride_n = (n - tile_n) // (num_tiles - 1)
    if (num_tiles - 1) * stride_n + tile_n != n:
        raise ValueError(
            f"Tile placement does not exactly cover n. "
            f"Got: (num_tiles-1)*stride_n + tile_n = {(num_tiles-1)*stride_n + tile_n}, expected n={n}. "
            f"Try changing --num_tiles."
        )
    if stride_n > tile_n:
        raise ValueError(
            f"stride_n={stride_n} > tile_n={tile_n} would create uncovered gaps. "
            f"Increase --num_tiles (or reduce width_ratio) so stride_n <= tile_n."
        )

    x_starts = [i * stride_n for i in range(num_tiles)]

    # ----------------- tiling along height (y) -----------------
    # CHANGED: number of rows is derived from height_ratio (2 => two rows)
    num_rows = args.num_rows if args.num_rows is not None else int(args.height_ratio)
    if num_rows < 2:
        # num_rows=1이면 y 타일링 없음(디버그 용)
        y_starts = [0]
    else:
        stride_m = (m - tile_m) // (num_rows - 1)

        # 정확 커버 강제(지금 x축에서 하던 방식과 동일)
        if (num_rows - 1) * stride_m + tile_m != m:
            raise ValueError(
                f"Row placement does not exactly cover m. "
                f"Got: (num_rows-1)*stride_m + tile_m = {(num_rows-1)*stride_m + tile_m}, expected m={m}. "
                f"Try changing --num_rows."
            )
        if stride_m > tile_m:
            raise ValueError(
                f"stride_m={stride_m} > tile_m={tile_m} would create uncovered gaps. "
                f"Increase --num_rows so stride_m <= tile_m."
            )

        y_starts = [j * stride_m for j in range(num_rows)]

    # ----------------- taper weights (2D) -----------------
    # CHANGED: use 2D weight (y and x)
    wx_1d = 0.5 * (1.0 - torch.cos(torch.linspace(0, math.pi, tile_n, device=model.device)))
    wy_1d = 0.5 * (1.0 - torch.cos(torch.linspace(0, math.pi, tile_m, device=model.device)))
    wx_1d = torch.clamp(wx_1d, min=1e-3).to(dtype=model.dtype)
    wy_1d = torch.clamp(wy_1d, min=1e-3).to(dtype=model.dtype)
    w2 = (wy_1d.view(1, tile_m, 1, 1) * wx_1d.view(1, 1, tile_n, 1)).contiguous()  # (1,tile_m,tile_n,1)

    # ----------------- ownership (x and y) -----------------
    # CHANGED: add owner_y (previously x-only)
    centers_x = torch.tensor([x0 + (tile_n - 1) / 2.0 for x0 in x_starts], device=model.device)
    xs = torch.arange(n, device=model.device).float()
    owner_x = torch.argmin((xs[:, None] - centers_x[None, :]).abs(), dim=1)  # (n,) -> tx

    centers_y = torch.tensor([y0 + (tile_m - 1) / 2.0 for y0 in y_starts], device=model.device)
    ys = torch.arange(m, device=model.device).float()
    owner_y = torch.argmin((ys[:, None] - centers_y[None, :]).abs(), dim=1)  # (m,) in [0..num_rows-1]

    # ----------------- tile -> region mapping (3 regions, based on x-tile index) -----------------
    num_regions = 3
    tile_to_region_x = [min(num_regions - 1, (tx * num_regions) // num_tiles) for tx in range(num_tiles)]

    # ----------------- build region-conditioned text caches -----------------
    clauses = [left_clause, mid_clause, right_clause]
    region_input_ids = []
    region_attention_masks = []
    region_past = []

    for rid in range(num_regions):
        extra = (" " + clauses[rid]) * max(0, int(args.clause_repeat))
        cond_prompt_r = f"Generate an image: {total_prompt}.{extra}"

        class_info_r = model.prepare_text_conditions(cond_prompt_r, args.cfg_prompt)
        input_ids_2 = class_info_r["input_ids"]  # [cond, uncond]
        attn_2 = class_info_r["attention_mask"]

        assert len(input_ids_2) == 2, "Expected [cond, uncond] prompts from prepare_text_conditions."

        input_ids_r, attn_r = expand_cfg_batch(input_ids_2, attn_2, base_bsz=base_bsz, cfg=args.cfg)

        try:
            past_r = model.prepare_past_key_values(input_ids=input_ids_r, attention_mask=attn_r)
        except TypeError:
            past_r = model.prepare_past_key_values(input_ids=input_ids_r)

        region_input_ids.append(input_ids_r)
        region_attention_masks.append(attn_r)
        region_past.append(past_r)

    bsz = region_attention_masks[0].shape[0]

    # ----------------- global tokens -----------------
    D = model.token_embed_dim
    global_tokens = torch.zeros(bsz, m, n, D, device=model.device, dtype=model.dtype)

    # ----------------- global mask + global perm (SSOT) -----------------
    global_mask = torch.ones(bsz, m, n, device=model.device, dtype=model.dtype)
    global_perm = make_global_perm(bsz, L, device=model.device)

    if args.cfg != 1.0:
        global_perm = global_perm.clone()
        global_perm[bsz // 2:] = global_perm[: bsz // 2]
        global_mask = global_mask.clone()
        global_mask[bsz // 2:] = global_mask[: bsz // 2]

    # CHANGED: dummy orders length uses tile_m*tile_n (not m*n)
    dummy_orders_tile = torch.zeros(bsz, tile_m * tile_n, device=model.device, dtype=torch.long)

    # ----------------- global-step loop -----------------
    for step in tqdm(range(args.num_iter), desc="Global-step", disable=False):
        num = torch.zeros_like(global_tokens)
        den = torch.zeros(bsz, m, n, 1, device=model.device, dtype=model.dtype)
        upd = torch.zeros(bsz, m, n, device=model.device, dtype=torch.bool)

        global_mask_flat = global_mask.view(bsz, L)

        if step >= args.num_iter - 1:
            global_mask_to_pred_flat = global_mask_flat.bool()
            global_mask_next_flat = torch.zeros_like(global_mask_flat)
        else:
            mask_ratio = math.cos(math.pi / 2.0 * (step + 1) / args.num_iter)
            target_len = int(math.floor(L * mask_ratio))

            unknown0 = int(global_mask_flat[0].sum().item())
            mask_len0 = max(1, min(unknown0 - 1, target_len))

            global_mask_next_flat = mask_from_perm_keep_last(global_perm, mask_len0, dtype=model.dtype)
            global_mask_to_pred_flat = (global_mask_flat.bool() ^ global_mask_next_flat.bool())

        if args.cfg != 1.0:
            global_mask_next_flat = global_mask_next_flat.clone()
            global_mask_next_flat[bsz // 2:] = global_mask_next_flat[: bsz // 2]
            global_mask_to_pred_flat = global_mask_to_pred_flat.clone()
            global_mask_to_pred_flat[bsz // 2:] = global_mask_to_pred_flat[: bsz // 2]

        global_mask_next = global_mask_next_flat.view(bsz, m, n)
        global_mask_to_pred = global_mask_to_pred_flat.view(bsz, m, n)

        # ----------------- 2D tile loop (CHANGED) -----------------
        for ty, y0 in enumerate(y_starts):
            y1 = y0 + tile_m
            for tx, x0 in enumerate(x_starts):
                x1 = x0 + tile_n

                rid = tile_to_region_x[tx]
                input_ids = region_input_ids[rid]
                attention_mask = region_attention_masks[rid]
                past = region_past[rid]

                tokens_in = global_tokens[:, y0:y1, x0:x1, :].contiguous()
                mask_in = global_mask[:, y0:y1, x0:x1].contiguous()

                m2p = global_mask_to_pred[:, y0:y1, x0:x1]  # bool

                ownx = (owner_x[x0:x1] == tx).view(1, 1, tile_n).expand(bsz, tile_m, tile_n)
                owny = (owner_y[y0:y1] == ty).view(1, tile_m, 1).expand(bsz, tile_m, tile_n)
                mask_to_pred_tile = (m2p & ownx & owny)

                if not mask_to_pred_tile.any():
                    continue

                mask_next_tile = global_mask_next[:, y0:y1, x0:x1].contiguous()

                out = model.sample_step_tokens(
                    step=step,
                    num_iter=args.num_iter,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tokens=tokens_in,
                    mask=mask_in,
                    orders=dummy_orders_tile,
                    past_key_values=past,
                    cfg=args.cfg,
                    cfg_schedule=args.cfg_schedule,
                    temperature=args.temperature,
                    image_shape=(tile_m, tile_n),
                    # requires modified sample_step_tokens
                    mask_to_pred=mask_to_pred_tile,
                    mask_next=mask_next_tile,
                )

                tokens_out = out["tokens"]

                w_upd = w2 * mask_to_pred_tile.unsqueeze(-1).to(dtype=model.dtype)

                num[:, y0:y1, x0:x1, :] += w_upd * tokens_out
                den[:, y0:y1, x0:x1, :] += w_upd
                upd[:, y0:y1, x0:x1] |= mask_to_pred_tile

        merged = num / (den + 1e-6)
        upd4 = upd.unsqueeze(-1).expand(-1, -1, -1, D)
        global_tokens = torch.where(upd4, merged, global_tokens)

        global_mask = global_mask_next

    # ----------------- decode panorama -----------------
    pano = model.decode(global_tokens)  # (bsz, 3, H, W)
    if args.cfg != 1.0:
        pano = pano[: bsz // 2]

    stem = out_path.stem
    save_tensor_image(pano[0], out_path)
    print(f"[OK] Saved canvas: {out_path}", flush=True)

    # ----------------- save per-view tiles (CHANGED: 2D) -----------------
    px_per_token = 16
    tile_w_px = tile_n * px_per_token
    tile_h_px = tile_m * px_per_token

    pad = int(args.pad_tokens)
    if pad < 0:
        raise ValueError("--pad_tokens must be >= 0")

    for ty, y0 in enumerate(y_starts):
        for tx, x0 in enumerate(x_starts):
            yL = max(0, y0 - pad)
            yR = min(m, y0 + tile_m + pad)
            xL = max(0, x0 - pad)
            xR = min(n, x0 + tile_n + pad)

            win_tokens = global_tokens[:, yL:yR, xL:xR, :].contiguous()
            win_img = model.decode(win_tokens)
            if args.cfg != 1.0:
                win_img = win_img[: bsz // 2]

            start_y_px = (y0 - yL) * px_per_token
            end_y_px = start_y_px + tile_h_px
            start_x_px = (x0 - xL) * px_per_token
            end_x_px = start_x_px + tile_w_px

            view_img = win_img[0, :, start_y_px:end_y_px, start_x_px:end_x_px]

            view_path = out_dir / f"{stem}_r{ty:02d}_c{tx:02d}.png"
            save_tensor_image(view_img, view_path)

    print(f"[OK] Saved {len(y_starts)*len(x_starts)} per-view tiles into: {out_dir}", flush=True)
