'''
text2pano_v2(251216)
- Text prompt의 형식을 지정하고, 그에 따라 num_tiles = 9로 두고 3개씩 분할해서 들어감
- Prompt 예시: "A beautiful beach on the left, a bustling city in the center, and a serene mountain on the right."
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
    parser.add_argument("--pan_ratio", type=int, default=4)
    parser.add_argument("--output", type=str, default="output.jpg")
    parser.add_argument("--num_tiles", type=int, default=5)
    parser.add_argument("--pad_tokens", type=int, default=16)  # padding context in latent tokens

    # naive clause weighting
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

    # base prompt (for reference / print)
    base_cond_prompt = f"Generate an image: {total_prompt}"
    print(base_cond_prompt, flush=True)

    base_bsz = args.grid_size ** 2

    # ----------------- panorama token grid shape -----------------
    m = args.image_size // 16
    n = (args.image_size * args.pan_ratio) // 16
    L = m * n

    # ----------------- tiling along width (n) -----------------
    tile_m = m
    tile_n = args.image_size // 16  # 32 tokens wide => 512 px

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
            f"Increase --num_tiles (or reduce pano width) so stride_n <= tile_n."
        )

    x_starts = [i * stride_n for i in range(num_tiles)]

    # taper weight across tile width
    w_1d = 0.5 * (1.0 - torch.cos(torch.linspace(0, math.pi, tile_n, device=model.device)))
    w_1d = torch.clamp(w_1d, min=1e-3)
    w = w_1d.view(1, 1, tile_n, 1).to(dtype=model.dtype)  # (1,1,tile_n,1)

    # ----------------- overlap ownership (x-wise) -----------------
    centers = torch.tensor([x0 + (tile_n - 1) / 2.0 for x0 in x_starts], device=model.device)  # (T,)
    xs = torch.arange(n, device=model.device).float()  # (n,)
    owner_x = torch.argmin((xs[:, None] - centers[None, :]).abs(), dim=1)  # (n,), int in [0..T-1]

    # ----------------- tile -> region mapping (3 regions) -----------------
    # For num_tiles=9: [0,0,0,1,1,1,2,2,2]
    num_regions = 3
    tile_to_region = [min(num_regions - 1, (t * num_regions) // num_tiles) for t in range(num_tiles)]

    # ----------------- build region-conditioned text caches -----------------
    # We do NOT require user to write 3 prompts; we auto-generate variants by repeating the target clause.
    clauses = [left_clause, mid_clause, right_clause]
    region_input_ids = []
    region_attention_masks = []
    region_past = []

    for rid in range(num_regions):
        extra = (" " + clauses[rid]) * max(0, int(args.clause_repeat))
        cond_prompt_r = f"Generate an image: {total_prompt}.{extra}"

        class_info_r = model.prepare_text_conditions(cond_prompt_r, args.cfg_prompt)
        input_ids_2 = class_info_r["input_ids"]         # [cond, uncond]
        attn_2 = class_info_r["attention_mask"]

        assert len(input_ids_2) == 2, "Expected [cond, uncond] prompts from prepare_text_conditions."

        input_ids_r, attn_r = expand_cfg_batch(input_ids_2, attn_2, base_bsz=base_bsz, cfg=args.cfg)

        # KV cache for this region (once)
        try:
            past_r = model.prepare_past_key_values(input_ids=input_ids_r, attention_mask=attn_r)
        except TypeError:
            past_r = model.prepare_past_key_values(input_ids=input_ids_r)

        region_input_ids.append(input_ids_r)
        region_attention_masks.append(attn_r)
        region_past.append(past_r)

    # bsz determined by expanded batch
    bsz = region_attention_masks[0].shape[0]

    # ----------------- global tokens -----------------
    D = model.token_embed_dim
    global_tokens = torch.zeros(bsz, m, n, D, device=model.device, dtype=model.dtype)

    # ----------------- global mask + global perm (SSOT) -----------------
    global_mask = torch.ones(bsz, m, n, device=model.device, dtype=model.dtype)
    global_perm = make_global_perm(bsz, L, device=model.device)

    # keep cond/uncond aligned
    if args.cfg != 1.0:
        global_perm = global_perm.clone()
        global_perm[bsz // 2:] = global_perm[: bsz // 2]
        global_mask = global_mask.clone()
        global_mask[bsz // 2:] = global_mask[: bsz // 2]

    # dummy orders to avoid consuming RNG inside sample_step_tokens when mask_to_pred is provided
    dummy_orders_tile = torch.zeros(bsz, tile_m * tile_n, device=model.device, dtype=torch.long)

    # ----------------- global-step loop -----------------
    for step in tqdm(range(args.num_iter), desc="Global-step", disable=False):
        num = torch.zeros_like(global_tokens)
        den = torch.zeros(bsz, m, n, 1, device=model.device, dtype=model.dtype)
        upd = torch.zeros(bsz, m, n, device=model.device, dtype=torch.bool)

        # ---- build global_mask_next / global_mask_to_pred (SSOT) ----
        global_mask_flat = global_mask.view(bsz, L)

        if step >= args.num_iter - 1:
            global_mask_to_pred_flat = global_mask_flat.bool()
            global_mask_next_flat = torch.zeros_like(global_mask_flat)  # everything becomes known
        else:
            mask_ratio = math.cos(math.pi / 2.0 * (step + 1) / args.num_iter)
            target_len = int(math.floor(L * mask_ratio))

            # mimic original safety: leave at least 1 masked for the NEXT iteration
            unknown0 = int(global_mask_flat[0].sum().item())
            mask_len0 = max(1, min(unknown0 - 1, target_len))

            global_mask_next_flat = mask_from_perm_keep_last(global_perm, mask_len0, dtype=model.dtype)
            global_mask_to_pred_flat = (global_mask_flat.bool() ^ global_mask_next_flat.bool())

        # keep cond/uncond aligned
        if args.cfg != 1.0:
            global_mask_next_flat = global_mask_next_flat.clone()
            global_mask_next_flat[bsz // 2:] = global_mask_next_flat[: bsz // 2]
            global_mask_to_pred_flat = global_mask_to_pred_flat.clone()
            global_mask_to_pred_flat[bsz // 2:] = global_mask_to_pred_flat[: bsz // 2]

        global_mask_next = global_mask_next_flat.view(bsz, m, n)
        global_mask_to_pred = global_mask_to_pred_flat.view(bsz, m, n)

        # ---- tile loop ----
        for t, x0 in enumerate(x_starts):
            x1 = x0 + tile_n

            rid = tile_to_region[t]
            input_ids = region_input_ids[rid]
            attention_mask = region_attention_masks[rid]
            past = region_past[rid]

            tokens_in = global_tokens[:, :, x0:x1, :].contiguous()   # (B, tile_m, tile_n, D)
            mask_in = global_mask[:, :, x0:x1].contiguous()          # (B, tile_m, tile_n)

            # global decides prediction positions; tile takes slice
            m2p = global_mask_to_pred[:, :, x0:x1]                   # (B, tile_m, tile_n) bool

            # ownership: only positions whose global x is owned by this tile can be predicted here
            own = (owner_x[x0:x1] == t).view(1, 1, tile_n).expand(bsz, tile_m, tile_n)
            mask_to_pred_tile = (m2p & own)

            if not mask_to_pred_tile.any():
                continue

            mask_next_tile = global_mask_next[:, :, x0:x1].contiguous()

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
                # external control (requires your modified sample_step_tokens)
                mask_to_pred=mask_to_pred_tile,   # (B, tile_m, tile_n) bool
                mask_next=mask_next_tile,         # (B, tile_m, tile_n) float 0/1
            )

            tokens_out = out["tokens"]  # (B, tile_m, tile_n, D)

            w_upd = w * mask_to_pred_tile.unsqueeze(-1).to(dtype=model.dtype)

            num[:, :, x0:x1, :] += w_upd * tokens_out
            den[:, :, x0:x1, :] += w_upd
            upd[:, :, x0:x1] |= mask_to_pred_tile

        merged = num / (den + 1e-6)
        upd4 = upd.unsqueeze(-1).expand(-1, -1, -1, D)
        global_tokens = torch.where(upd4, merged, global_tokens)

        global_mask = global_mask_next

    # ----------------- decode panorama -----------------
    pano = model.decode(global_tokens)  # (bsz, 3, H, W)
    if args.cfg != 1.0:
        pano = pano[: bsz // 2]  # keep conditional half

    # ----------------- save panorama -----------------
    stem = out_path.stem
    save_tensor_image(pano[0], out_path)
    print(f"[OK] Saved panorama: {out_path}", flush=True)

    # ----------------- save per-view FINAL images (NOT pano crops) -----------------
    px_per_token = 16
    tile_w_px = tile_n * px_per_token
    tile_h_px = tile_m * px_per_token

    pad = int(args.pad_tokens)
    if pad < 0:
        raise ValueError("--pad_tokens must be >= 0")

    for t, x0 in enumerate(x_starts):
        xL = max(0, x0 - pad)
        xR = min(n, x0 + tile_n + pad)

        win_tokens = global_tokens[:, :, xL:xR, :].contiguous()
        win_img = model.decode(win_tokens)
        if args.cfg != 1.0:
            win_img = win_img[: bsz // 2]

        start_px = (x0 - xL) * px_per_token
        end_px = start_px + tile_w_px
        view_img = win_img[0, :, :tile_h_px, start_px:end_px]

        view_path = out_dir / f"{stem}_view{t:02d}.png"
        save_tensor_image(view_img, view_path)

    print(f"[OK] Saved {num_tiles} per-view images (padded-latent decode) into: {out_dir}", flush=True)
