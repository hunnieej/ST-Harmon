import argparse
import math
from pathlib import Path
import torch
from mmengine.config import Config
from PIL import Image
from tqdm import tqdm
from src.builder import BUILDER
import yaml

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
    # argsort(rand) gives a permutation of indices
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file path.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a dog on the left and a cat on the right.")
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
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    out_path = Path(args.output).resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "log.yaml"
    with open(log_path, "w", encoding="utf-8") as f:
        yaml.dump(
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

    # ----------------- prompts -----------------
    args.prompt = f"Generate an image: {args.prompt}"
    print(args.prompt, flush=True)
    class_info = model.prepare_text_conditions(args.prompt, args.cfg_prompt)
    input_ids = class_info["input_ids"]
    attention_mask = class_info["attention_mask"]
    assert len(input_ids) == 2, "Expected [cond, uncond] prompts from prepare_text_conditions."

    if args.cfg == 1.0:
        input_ids = input_ids[:1]
        attention_mask = attention_mask[:1]

    base_bsz = args.grid_size ** 2

    if args.cfg != 1.0:
        input_ids = torch.cat([
            input_ids[:1].expand(base_bsz, -1),   # conditional
            input_ids[1:].expand(base_bsz, -1),   # unconditional
        ])
        attention_mask = torch.cat([
            attention_mask[:1].expand(base_bsz, -1),
            attention_mask[1:].expand(base_bsz, -1),
        ])
    else:
        input_ids = input_ids.expand(base_bsz, -1)
        attention_mask = attention_mask.expand(base_bsz, -1)

    bsz = attention_mask.shape[0]

    # ----------------- panorama token grid shape -----------------
    # 512x2048 => latent grid (m,n) = (32, 128) if patch/stride=16
    m = args.image_size // 16
    n = (args.image_size * args.pan_ratio) // 16  # 2048//16 = 128
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

    # taper weight across tile width (kept; with ownership it cancels out, but harmless)
    w_1d = 0.5 * (1.0 - torch.cos(torch.linspace(0, math.pi, tile_n, device=model.device)))
    w_1d = torch.clamp(w_1d, min=1e-3)
    w = w_1d.view(1, 1, tile_n, 1).to(dtype=model.dtype)  # (1,1,tile_n,1)

    # ----------------- overlap ownership (x-wise) -----------------
    # assign each global x (0..n-1) to exactly one tile based on nearest tile center
    centers = torch.tensor([x0 + (tile_n - 1) / 2.0 for x0 in x_starts], device=model.device)  # (T,)
    xs = torch.arange(n, device=model.device).float()  # (n,)
    owner_x = torch.argmin((xs[:, None] - centers[None, :]).abs(), dim=1)  # (n,), int in [0..T-1]

    # ----------------- global tokens -----------------
    D = model.token_embed_dim
    global_tokens = torch.zeros(bsz, m, n, D, device=model.device, dtype=model.dtype)

    # ----------------- global mask + global perm (SSOT) -----------------
    # global_mask: 1=unknown, 0=known
    global_mask = torch.ones(bsz, m, n, device=model.device, dtype=model.dtype)

    # global_perm: (B, L) indices in prediction order (early -> late)
    global_perm = make_global_perm(bsz, L, device=model.device)

    # keep cond/uncond aligned
    if args.cfg != 1.0:
        global_perm = global_perm.clone()
        global_perm[bsz // 2:] = global_perm[: bsz // 2]
        global_mask = global_mask.clone()
        global_mask[bsz // 2:] = global_mask[: bsz // 2]

    # dummy orders to avoid consuming RNG inside sample_step_tokens when mask_to_pred is provided
    dummy_orders_tile = torch.zeros(bsz, tile_m * tile_n, device=model.device, dtype=torch.long)

    # ----------------- KV cache (once) -----------------
    try:
        past = model.prepare_past_key_values(input_ids=input_ids, attention_mask=attention_mask)
    except TypeError:
        past = model.prepare_past_key_values(input_ids=input_ids)

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

            global_mask_next_flat = mask_from_perm_keep_last(
                global_perm, mask_len0, dtype=model.dtype
            )
            global_mask_to_pred_flat = (global_mask_flat.bool() ^ global_mask_next_flat.bool())

        # keep cond/uncond aligned
        if args.cfg != 1.0:
            global_mask_next_flat = global_mask_next_flat.clone()
            global_mask_next_flat[bsz // 2:] = global_mask_next_flat[: bsz // 2]
            global_mask_to_pred_flat = global_mask_to_pred_flat.clone()
            global_mask_to_pred_flat[bsz // 2:] = global_mask_to_pred_flat[: bsz // 2]

        global_mask_next = global_mask_next_flat.view(bsz, m, n)
        global_mask_to_pred = global_mask_to_pred_flat.view(bsz, m, n)

        # ---- tile loop: slice tokens + slice global mask + slice global mask_to_pred ----
        for t, x0 in enumerate(x_starts):
            x1 = x0 + tile_n

            tokens_in = global_tokens[:, :, x0:x1, :].contiguous()           # (B, tile_m, tile_n, D)
            mask_in = global_mask[:, :, x0:x1].contiguous()                  # (B, tile_m, tile_n)

            # global decides prediction positions; tile takes slice
            m2p = global_mask_to_pred[:, :, x0:x1]                           # (B, tile_m, tile_n) bool

            # ownership: only positions whose global x is owned by this tile can be predicted here
            own = (owner_x[x0:x1] == t).view(1, 1, tile_n).expand(bsz, tile_m, tile_n)
            mask_to_pred_tile = (m2p & own)

            # if this tile predicts nothing this step, skip compute
            if not mask_to_pred_tile.any():
                continue

            # optional: pass the next mask slice so returned mask matches SSOT
            mask_next_tile = global_mask_next[:, :, x0:x1].contiguous()

            out = model.sample_step_tokens(
                step=step,
                num_iter=args.num_iter,
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokens=tokens_in,
                mask=mask_in,
                orders=dummy_orders_tile,              # unused in external-control mode, avoids RNG consumption
                past_key_values=past,
                cfg=args.cfg,
                cfg_schedule=args.cfg_schedule,
                temperature=args.temperature,
                image_shape=(tile_m, tile_n),
                # NEW: external control
                mask_to_pred=mask_to_pred_tile,        # (B, tile_m, tile_n) bool
                mask_next=mask_next_tile,              # (B, tile_m, tile_n) float 0/1
            )

            tokens_out = out["tokens"]  # (B, tile_m, tile_n, D)

            # only merge positions updated in THIS step (ownership already enforces uniqueness)
            w_upd = w * mask_to_pred_tile.unsqueeze(-1).to(dtype=model.dtype)

            num[:, :, x0:x1, :] += w_upd * tokens_out
            den[:, :, x0:x1, :] += w_upd
            upd[:, :, x0:x1] |= mask_to_pred_tile

        merged = num / (den + 1e-6)
        upd4 = upd.unsqueeze(-1).expand(-1, -1, -1, D)
        global_tokens = torch.where(upd4, merged, global_tokens)

        # advance SSOT mask
        global_mask = global_mask_next

    # ----------------- decode panorama -----------------
    pano = model.decode(global_tokens)  # (bsz, 3, 512, 2048)
    if args.cfg != 1.0:
        pano = pano[: bsz // 2]  # keep conditional half

    # ----------------- save panorama -----------------
    out_path = Path(args.output).resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem

    save_tensor_image(pano[0], out_path)
    print(f"[OK] Saved panorama: {out_path}", flush=True)

    # ----------------- save per-view FINAL images (NOT pano crops) -----------------
    # decode each view from a padded latent window, then crop center region in pixel space
    px_per_token = 16
    tile_w_px = tile_n * px_per_token  # 512
    tile_h_px = tile_m * px_per_token  # 512

    pad = int(args.pad_tokens)
    if pad < 0:
        raise ValueError("--pad_tokens must be >= 0")

    for t, x0 in enumerate(x_starts):
        xL = max(0, x0 - pad)
        xR = min(n, x0 + tile_n + pad)

        # decode padded latent window
        win_tokens = global_tokens[:, :, xL:xR, :].contiguous()   # (bsz, 32, win_n, D)
        win_img = model.decode(win_tokens)                        # (bsz, 3, 512, win_w_px)
        if args.cfg != 1.0:
            win_img = win_img[: bsz // 2]                         # keep conditional half

        # crop out the original tile region from the decoded padded window
        start_px = (x0 - xL) * px_per_token
        end_px = start_px + tile_w_px
        view_img = win_img[0, :, :tile_h_px, start_px:end_px]     # (3, 512, 512)

        view_path = out_dir / f"{stem}_view{t:02d}.png"
        save_tensor_image(view_img, view_path)

    print(f"[OK] Saved {num_tiles} per-view images (padded-latent decode) into: {out_dir}", flush=True)
