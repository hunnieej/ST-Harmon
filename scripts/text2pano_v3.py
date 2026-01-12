"""
Date : 2025-12-22
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
from scripts.cylindrical_functions import precompute_cyl_yaw_inverse_writer
from scripts.importance_functions import (compute_global_cfg_importance_map,
                                          make_global_perm_from_score_masked, 
                                          save_importance_heatmap, 
                                          save_rank_map, 
                                          save_step_pred_mask)
from scripts.main_functions import (save_tensor_image, 
                                    expand_cfg_batch, 
                                    make_global_perm, 
                                    mask_from_perm_keep_last)

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
    parser.add_argument("--num_tiles", type=int, default=9)
    parser.add_argument("--fov_x_deg", type=float, default=80.0)
    parser.add_argument("--fov_y_deg", type=float, default=80.0)
    parser.add_argument("--save_steps", default="True", help="whether to save intermediate steps.")
    parser.add_argument("--importance", default="True", help="whether to use importance-based permutation.")
    parser.add_argument("--step_importance_mode", type=str, default="l2", choices=["l1","l2","linf","cos","rel_l2","rel_l1","dot"], help="mode for step-wise importance adjustment.")
    parser.add_argument("--step_importance_reduce", type=str, default="mean", choices=["mean", "median", "max"], help="reduction method for step-wise importance adjustment.")
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
    delta_max = math.pi / args.num_tiles
    tile_m = int(math.ceil(m / math.cos(delta_max)))   # e.g., m=32, T=9 -> 34
    tile_n = args.image_size // 16
    M = tile_m * tile_n
    dummy_orders_tile = torch.zeros(bsz, M, device=device, dtype=torch.long)


    # ----------------- precompute inverse-writer maps -----------------
    theta0, owner_x, _, read_lin_list, writer_lin_list, writer_pix_list = precompute_cyl_yaw_inverse_writer(
        m=m,
        n=n,
        tile_m=tile_m,
        tile_n=tile_n,
        num_tiles=args.num_tiles,
        fov_x_deg=args.fov_x_deg,
        fov_y_deg=args.fov_y_deg,
        device=device,
    )
    print("[OK] Precomputed inverse-writer yaw maps (no missing writers by construction).", flush=True)

    # ----------------- global tokens -----------------
    D = model.token_embed_dim
    global_tokens = torch.zeros(bsz, m, n, D, device=device, dtype=model.dtype)

    # ----------------- global mask + global perm (SSOT schedule on GLOBAL) -----------------
    global_mask = torch.ones(bsz, m, n, device=device, dtype=model.dtype)  # 1=unknown

    # ----------------- KV cache (once) -----------------
    try:
        past = model.prepare_past_key_values(input_ids=input_ids, attention_mask=attention_mask)
    except TypeError:
        past = model.prepare_past_key_values(input_ids=input_ids)

    # A-plan: text-only importance-based permutation
    if args.importance=="True":
        importance_map = compute_global_cfg_importance_map(
            model=model,
            step=0,
            num_iter=args.num_iter,
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            global_tokens=global_tokens,
            global_mask=global_mask,
            read_lin_list=read_lin_list,
            writer_lin_list=writer_lin_list,
            writer_pix_list=writer_pix_list,
            tile_m=tile_m,
            tile_n=tile_n,
            orders_tile=dummy_orders_tile,
            temperature=args.temperature,
            device=device,
            score_mode=args.step_importance_mode,
            batch_reduce=args.step_importance_reduce,
        )
        global_perm = make_global_perm_from_score_masked(
            score_map=importance_map,
            global_mask=global_mask,
            noise_std=0.01,
        )
        save_importance_heatmap(importance_map, out_dir / "importance_heatmap.png")
    else:
        global_perm = make_global_perm(bsz, L, device=device)
    save_rank_map(global_perm, m, n, out_dir / "rank_map.png")

    # keep cond/uncond aligned
    if args.cfg != 1.0:
        global_perm = global_perm.clone()
        global_perm[bsz // 2:] = global_perm[: bsz // 2]
        global_mask = global_mask.clone()
        global_mask[bsz // 2:] = global_mask[: bsz // 2]

    dummy_orders_tile = torch.zeros(bsz, M, device=device, dtype=torch.long)

    # ----------------- per step decoding -----------------
    @torch.inference_mode()
    def decode_and_save_step(step_idx: int, tokens_4d: torch.Tensor):
        tok = tokens_4d[:1].detach().contiguous().clone()
        img = model.decode(tok)  # (1, 3, Hpx, Wpx)
        img = torch.nan_to_num(img, nan=-1.0, posinf=1.0, neginf=-1.0)
        save_tensor_image(img[0], steps_dir / f"step{step_idx:02d}.png")
    
    if args.save_steps=="True":
        stem = out_path.stem
        steps_dir = out_dir / "steps"
        steps_dir.mkdir(parents=True, exist_ok=True)
        decode_and_save_step(0, global_tokens)

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
        if args.save_steps== "True":
            save_step_pred_mask(global_mask_to_pred_flat, m, n, steps_dir / f"predmask{step+1:02d}.png")
            # save_importance_heatmap(importance_map, steps_dir / f"importance{step+1:02d}.png")
            decode_and_save_step(step + 1, global_tokens)

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