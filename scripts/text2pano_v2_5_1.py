"""
scripts.text2pano_v2.5.1
251218ver (cyl-yaw inverse-writer)

Goal:
- Global latent is defined on the cylinder surface (unwrapped ERP grid: m x n).
- We render/generate tangent views (tiles) at different yaw centers (theta0[t]).
- Each step:
    global -> per-tile gather (view tokens)
    per-tile one-step unmasking (MAR step, externally controlled mask)
    merge back to global (SSOT writer per global cell)
- Y axis is fixed (row-wise identity): tile covers full cylinder height.
  This matches the "Y is fixed so full vertical extent is always included" requirement.
"""

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


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    # map angle to (-pi, pi]
    twopi = 2.0 * math.pi
    return (x + math.pi) % twopi - math.pi


def _assign_unique_int_indices(i_float: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """
    Greedy unique integer assignment for a 1D set of targets.

    i_float: (K,) float target positions
    returns: (K,) long assigned unique ints in [lo, hi]
    """
    K = int(i_float.numel())
    order = torch.argsort(i_float)  # stable ordering by target
    assigned = torch.full((K,), -1, device=i_float.device, dtype=torch.long)
    used = torch.zeros((hi - lo + 1,), device=i_float.device, dtype=torch.bool)

    for idx in order.tolist():
        tgt = float(i_float[idx].item())
        base = int(round(tgt))
        base = max(lo, min(hi, base))
        if not used[base - lo]:
            assigned[idx] = base
            used[base - lo] = True
            continue

        # search nearest free slot
        found = False
        for d in range(1, hi - lo + 2):
            left = base - d
            right = base + d
            if left >= lo and not used[left - lo]:
                assigned[idx] = left
                used[left - lo] = True
                found = True
                break
            if right <= hi and not used[right - lo]:
                assigned[idx] = right
                used[right - lo] = True
                found = True
                break
        if not found:
            raise ValueError("[assign_unique] No available integer slot. Increase tile_n or reduce owned columns.")

    return assigned


@torch.no_grad()
def precompute_cyl_yaw_inverse_writer(
    *,
    m: int,
    n: int,
    tile_m: int,
    tile_n: int,
    num_tiles: int,
    fov_x_deg: float,
    device: torch.device,
):
    """
    Inverse-writer cylindrical yaw mapping (Y fixed / full-height):

    Global (unwrapped cylinder) grid:
      - x in [0..n-1] corresponds to theta_x = 2pi*(x+0.5)/n
      - y in [0..m-1] is "height axis" and is used as identity into tile rows

    Tiles:
      - tile t has yaw center theta0[t] = 2pi*t/T
      - tile plane x coordinate is u in [-1,1] with FoV_x pinhole:
            delta = wrap(theta_x - theta0[t])
            delta = atan( u * tan(FoVx/2) )
        inverse:
            u = tan(delta) / tan(FoVx/2)

    We build:
      1) owner_x[x] = nearest tile center to theta_x (circular distance)
      2) For each tile:
            - collect owned global columns xs
            - compute target i_float from inverse u
            - assign UNIQUE integer i per owned x (injective) to avoid column collisions
      3) writer mapping (STRICT SSOT):
            Each global cell (y,x) writes from exactly one tile pixel:
                tile = owner_x[x]
                pix = y*tile_n + i_assigned_for_x_in_that_tile

    Also build a "read map" (global -> tile gather):
      - start from forward camera columns for context
      - override writer columns so that read_x_idx[i_assigned] == x (alignment)
      - y is identity: row y reads global row y

    Returns:
      theta0:            (T,) float
      owner_x:           (n,) long
      read_lin_list:     list[T] of (tile_m*tile_n,) long  (global lin per tile pixel, for gathering)
      writer_lin_list:   list[T] of (tile_m*owned_x_count,) long (global lin to be written by this tile)
      writer_pix_list:   list[T] of (tile_m*owned_x_count,) long (tile pixel indices corresponding to writer_lin)
    """
    T = int(num_tiles)
    if T < 2:
        raise ValueError("--num_tiles must be >= 2")
    if tile_m != m:
        raise ValueError("[cyl-yaw] This version requires tile_m == m (full height). Set tile_m=m.")
    if tile_n <= 0:
        raise ValueError("--tile_n must be > 0")

    fovx = math.radians(float(fov_x_deg))
    if not (0.0 < fovx < math.pi):
        raise ValueError("--fov_x_deg must be in (0, 180)")

    # coverage requirement under nearest-center ownership:
    # need FoV_x/2 >= pi/T, otherwise boundary columns cannot be seen by their owner tile at all.
    if (fovx / 2.0) + 1e-8 < (math.pi / T):
        raise ValueError(
            f"[cyl-yaw] FoV_x too small for num_tiles with nearest-center ownership. "
            f"Need FoV_x/2 >= pi/T. Got FoV_x/2={fovx/2:.4f}, pi/T={math.pi/T:.4f}. "
            f"Increase --fov_x_deg or decrease --num_tiles."
        )

    # tile yaw centers
    theta0 = torch.linspace(0.0, 2.0 * math.pi, steps=T + 1, device=device)[:-1]  # (T,)

    # global x centers in angle
    x = torch.arange(n, device=device).float()
    theta_x = 2.0 * math.pi * (x + 0.5) / float(n)  # (n,)

    # owner_x: nearest tile center
    # dist[t,x] = |wrap(theta_x - theta0[t])|
    dist = torch.abs(wrap_to_pi(theta_x.view(1, n) - theta0.view(T, 1)))  # (T,n)
    owner_x = torch.argmin(dist, dim=0).long()  # (n,)

    # forward columns for read context (per tile)
    i = torch.arange(tile_n, device=device).float()
    u = 2.0 * (i + 0.5) / float(tile_n) - 1.0  # [-1,1]
    alpha = torch.atan(u * math.tan(fovx / 2.0))  # (tile_n,)

    read_lin_list = []
    writer_lin_list = []
    writer_pix_list = []

    ys = torch.arange(m, device=device, dtype=torch.long).view(m, 1)  # (m,1)

    for t in range(T):
        # owned global x columns for this tile
        xs_owned = (owner_x == t).nonzero(as_tuple=True)[0].long()  # (Kx,)
        Kx = int(xs_owned.numel())
        if Kx == 0:
            raise ValueError(f"[cyl-yaw] tile {t} owns zero columns. Check n/num_tiles settings.")

        # delta angles for owned columns
        delta = wrap_to_pi(theta_x.index_select(0, xs_owned) - theta0[t])  # (Kx,)
        if torch.max(torch.abs(delta)).item() > (fovx / 2.0 + 1e-6):
            raise ValueError(
                f"[cyl-yaw] Owner tile {t} has a column outside FoV_x. "
                f"max|delta|={torch.max(torch.abs(delta)).item():.6f}, FoV_x/2={fovx/2:.6f}. "
                f"Increase FoV_x or adjust num_tiles."
            )

        # inverse mapping to tile u coordinate, then to i_float
        u_x = torch.tan(delta) / math.tan(fovx / 2.0)  # (Kx,)
        u_x = torch.clamp(u_x, -0.999, 0.999)
        i_float = (u_x + 1.0) * 0.5 * float(tile_n) - 0.5  # (Kx,)

        # assign UNIQUE integer i per owned x
        i_assigned = _assign_unique_int_indices(i_float, lo=0, hi=tile_n - 1)  # (Kx,) long

        # build read_x_idx for this tile:
        #   start with forward mapping (context),
        #   then override writer columns so writer pixels read the correct global x.
        theta_cols = torch.remainder(theta0[t] + alpha, 2.0 * math.pi)  # (tile_n,)
        x_float_fwd = theta_cols / (2.0 * math.pi) * float(n)           # [0,n)
        x_idx_fwd = torch.remainder(torch.round(x_float_fwd - 0.5).long(), n)  # (tile_n,)

        read_x_idx = x_idx_fwd.clone()
        read_x_idx.index_copy_(0, i_assigned, xs_owned)  # enforce alignment for writer columns

        # per-pixel read linear indices (m x tile_n)
        lin_read = (ys * n + read_x_idx.view(1, tile_n)).reshape(-1).long()  # (M,)
        read_lin_list.append(lin_read)

        # writer linear indices and writer pixel indices (strict SSOT)
        # global lin: (y,x)
        lin_writer = (ys * n + xs_owned.view(1, Kx)).reshape(-1).long()  # (m*Kx,)
        pix_writer = (ys * tile_n + i_assigned.view(1, Kx)).reshape(-1).long()  # (m*Kx,)

        writer_lin_list.append(lin_writer)
        writer_pix_list.append(pix_writer)

    # final sanity: all global cells must be covered exactly once
    all_lin = torch.cat(writer_lin_list, dim=0)
    if int(all_lin.numel()) != int(m * n):
        raise ValueError(f"[cyl-yaw] writer coverage mismatch: got {all_lin.numel()} vs expected {m*n}")
    # uniqueness check
    uniq = torch.unique(all_lin)
    if int(uniq.numel()) != int(m * n):
        raise ValueError("[cyl-yaw] writer_lin has duplicates. This should not happen with owner_x partition.")

    return theta0, owner_x, read_lin_list, writer_lin_list, writer_pix_list


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
    parser.add_argument("--height_ratio", type=int, default=1)
    parser.add_argument("--width_ratio", type=int, default=4)

    parser.add_argument("--output", type=str, default="output.jpg")

    # yaw tiles around cylinder
    parser.add_argument("--num_tiles", type=int, default=9)
    parser.add_argument("--fov_x_deg", type=float, default=80.0)

    # kept for compatibility / future (not used in yaw+Y-fixed version)
    parser.add_argument("--pad_tokens", type=int, default=0)

    args = parser.parse_args()
    torch.set_grad_enabled(False)

    # ----------------- output / log.yaml -----------------
    out_path = Path(args.output).resolve()
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "log.yaml"
    with open(log_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    print(f"[OK] Saved log: {log_path}", flush=True)

    # ----------------- build model -----------------
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).eval().cuda()
    model = model.to(model.dtype)

    checkpoint = torch.load(args.checkpoint, map_location="cuda")
    _ = model.load_state_dict(checkpoint, strict=False)

    device = model.device

    # ----------------- prompts (GLOBAL only) -----------------
    total_prompt = args.prompt
    cond_prompt = f"Generate an image: {total_prompt}"
    class_info = model.prepare_text_conditions(cond_prompt, args.cfg_prompt)
    input_ids_2 = class_info["input_ids"]          # [cond, uncond]
    attn_2 = class_info["attention_mask"]
    assert len(input_ids_2) == 2, "Expected [cond, uncond] prompts."

    base_bsz = args.grid_size ** 2
    input_ids, attention_mask = expand_cfg_batch(input_ids_2, attn_2, base_bsz=base_bsz, cfg=args.cfg)
    bsz = attention_mask.shape[0]

    # ----------------- global token grid shape -----------------
    m = (args.image_size * args.height_ratio) // 16
    n = (args.image_size * args.width_ratio) // 16
    L = m * n

    # tile shape: full height, fixed width=512px in tokens
    tile_m = m
    tile_n = args.image_size // 16
    M = tile_m * tile_n

    # ----------------- precompute inverse-writer maps -----------------
    theta0, owner_x, read_lin_list, writer_lin_list, writer_pix_list = precompute_cyl_yaw_inverse_writer(
        m=m,
        n=n,
        tile_m=tile_m,
        tile_n=tile_n,
        num_tiles=args.num_tiles,
        fov_x_deg=args.fov_x_deg,
        device=device,
    )
    print("[OK] Precomputed inverse-writer yaw maps (no missing writers by construction).", flush=True)

    # ----------------- global tokens -----------------
    D = model.token_embed_dim
    global_tokens = torch.zeros(bsz, m, n, D, device=device, dtype=model.dtype)

    # ----------------- global mask + global perm (SSOT schedule on GLOBAL) -----------------
    global_mask = torch.ones(bsz, m, n, device=device, dtype=model.dtype)  # 1=unknown
    global_perm = make_global_perm(bsz, L, device=device)

    # keep cond/uncond aligned
    if args.cfg != 1.0:
        global_perm = global_perm.clone()
        global_perm[bsz // 2:] = global_perm[: bsz // 2]
        global_mask = global_mask.clone()
        global_mask[bsz // 2:] = global_mask[: bsz // 2]

    dummy_orders_tile = torch.zeros(bsz, M, device=device, dtype=torch.long)

    # ----------------- KV cache (once) -----------------
    try:
        past = model.prepare_past_key_values(input_ids=input_ids, attention_mask=attention_mask)
    except TypeError:
        past = model.prepare_past_key_values(input_ids=input_ids)

    # ----------------- global-step loop -----------------
    for step in tqdm(range(args.num_iter), desc="Global-step", disable=False):
        global_tokens_flat = global_tokens.view(bsz, L, D)
        global_mask_flat = global_mask.view(bsz, L)

        # buffers to accumulate updated global cells only (flat)
        num_flat = torch.zeros_like(global_tokens_flat)
        den_flat = torch.zeros(bsz, L, 1, device=device, dtype=model.dtype)
        upd_int_flat = torch.zeros(bsz, L, device=device, dtype=model.dtype)

        # ---- build global_mask_next / global_mask_to_pred (GLOBAL SSOT schedule) ----
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

        # ---- tile loop ----
        for t in range(args.num_tiles):
            lin_read = read_lin_list[t]     # (M,)
            lin_write = writer_lin_list[t]  # (K,)
            pix_write = writer_pix_list[t]  # (K,)
            K = int(lin_write.numel())

            # gather view tokens/masks from global (context)
            tokens_in_flat = global_tokens_flat.index_select(dim=1, index=lin_read)  # (B,M,D)
            mask_in_flat = global_mask_flat.index_select(dim=1, index=lin_read)      # (B,M)

            # figure out which GLOBAL cells owned by this tile are predicted this step
            m2p_writer = global_mask_to_pred_flat.index_select(dim=1, index=lin_write)  # (B,K) bool
            if not m2p_writer.any():
                continue

            # build tile-level mask_to_pred / mask_next (pixel space)
            mask_to_pred_flat = torch.zeros(bsz, M, device=device, dtype=torch.bool)
            mask_to_pred_flat[:, pix_write] = m2p_writer

            mask_next_flat = mask_in_flat.clone()
            next_writer = global_mask_next_flat.index_select(dim=1, index=lin_write)  # (B,K) float 0/1
            mask_next_flat[:, pix_write] = next_writer

            # reshape to (B, tile_m, tile_n, ...)
            tokens_in = tokens_in_flat.view(bsz, tile_m, tile_n, D)
            mask_in = mask_in_flat.view(bsz, tile_m, tile_n)
            mask_to_pred_tile = mask_to_pred_flat.view(bsz, tile_m, tile_n)
            mask_next_tile = mask_next_flat.view(bsz, tile_m, tile_n)

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
                # external control
                mask_to_pred=mask_to_pred_tile,
                mask_next=mask_next_tile,
            )

            tokens_out_flat = out["tokens"].view(bsz, M, D)  # (B,M,D)

            # merge back ONLY for writer-owned global cells
            tokens_writer = tokens_out_flat[:, pix_write, :]  # (B,K,D)

            w = m2p_writer.unsqueeze(-1).to(dtype=model.dtype)  # (B,K,1)

            idx_D = lin_write.view(1, K, 1).expand(bsz, K, D)
            idx_1 = lin_write.view(1, K, 1).expand(bsz, K, 1)
            idx_u = lin_write.view(1, K).expand(bsz, K)

            num_flat.scatter_add_(dim=1, index=idx_D, src=w * tokens_writer)
            den_flat.scatter_add_(dim=1, index=idx_1, src=w)
            upd_int_flat.scatter_add_(dim=1, index=idx_u, src=m2p_writer.to(dtype=model.dtype))

        # apply updates
        upd_flat = upd_int_flat > 0
        merged_flat = num_flat / (den_flat + 1e-6)
        global_tokens_flat = torch.where(upd_flat.unsqueeze(-1), merged_flat, global_tokens_flat)

        # restore shape + advance global mask
        global_tokens = global_tokens_flat.view(bsz, m, n, D)
        global_mask = global_mask_next_flat.view(bsz, m, n)

    # ----------------- decode cylinder texture (unwrapped ERP) -----------------
    pano = model.decode(global_tokens)  # (bsz, 3, Hpx, Wpx)
    if args.cfg != 1.0:
        pano = pano[: bsz // 2]  # keep conditional half

    save_tensor_image(pano[0], out_path)
    print(f"[OK] Saved cylinder texture (unwrapped ERP): {out_path}", flush=True)

    # ----------------- optional: save each tile view (debug) -----------------
    stem = out_path.stem
    global_tokens_flat = global_tokens.view(bsz, L, D)

    for t in range(args.num_tiles):
        lin_read = read_lin_list[t]
        view_tokens_flat = global_tokens_flat.index_select(dim=1, index=lin_read).contiguous()  # (B,M,D)
        view_tokens = view_tokens_flat.view(bsz, tile_m, tile_n, D)
        view_img = model.decode(view_tokens)
        if args.cfg != 1.0:
            view_img = view_img[: bsz // 2]
        view_path = out_dir / f"{stem}_tile{t:02d}.png"
        save_tensor_image(view_img[0], view_path)

    print(f"[OK] Saved {args.num_tiles} tile views into: {out_dir}", flush=True)
