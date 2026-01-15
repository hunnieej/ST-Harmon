# import torch
# import torch.nn.functional as F
# from PIL import Image
# from mmengine.config import Config
# import argparse
# import numpy as np
# import os
# from src.builder import BUILDER

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('config', help='config file path.')
#     parser.add_argument("--checkpoint", type=str, required=True)
#     parser.add_argument("--prompt", type=str, default='a panoramic view of a living room')
#     parser.add_argument("--num_views", type=int, default=4, help="Number of sliding windows")
#     parser.add_argument("--image_size", type=int, default=512)
#     parser.add_argument("--overlap_ratio", type=float, default=0.25)
#     parser.add_argument("--cfg", type=float, default=3.0)
#     parser.add_argument("--output", type=str, default='outputs/result.jpg')
#     args = parser.parse_args()

#     # 1. Load Model
#     config = Config.fromfile(args.config)
#     model = BUILDER.build(config.model).eval().cuda().to(torch.bfloat16)
#     checkpoint = torch.load(args.checkpoint)
#     model.load_state_dict(checkpoint, strict=False)

#     # 2. Text Condition
#     full_prompt = f"Generate an image: {args.prompt}"
#     print(f"Prompt: {full_prompt}")
#     class_info = model.prepare_text_conditions(full_prompt, 'Generate an image.')

#     output_dir = os.path.dirname(args.output)
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)

#     generated_images = [] 
#     last_tokens = None # (B, m, n, d) - 이전 뷰의 Latent 저장소

#     for i in range(args.num_views):
#         print(f"\n>>> Generating View {i+1}/{args.num_views}...")

#         # --- A. Prepare Force Tokens (Latent Slicing + Erosion) ---
#         force_tokens = None
#         force_mask = None

#         if i > 0 and last_tokens is not None:
#             # last_tokens: (B, m, n, d)
#             B, m, n, d = last_tokens.shape
            
#             # 1. Overlap 너비 계산 (토큰 단위)
#             latent_overlap_w = int(n * args.overlap_ratio)
            
#             # 2. 토큰 복사 (Latent Stitching)
#             # 이전 뷰의 오른쪽 끝 토큰을 가져옴
#             overlap_tokens = last_tokens[:, :, -latent_overlap_w:, :]
            
#             # 현재 뷰의 왼쪽 끝에 심기
#             force_tokens = torch.zeros_like(last_tokens) # 나머지는 0 (의미 없음, 마스크로 가려짐)
#             force_tokens[:, :, :latent_overlap_w, :] = overlap_tokens
            
#             # 3. Mask Erosion (경계 허물기)
#             # 이론대로 "연결 부위"는 강제하지 않고 모델이 다시 그리게 함
#             # Positional Embedding 충돌 완화 목적
#             erosion_margin = 2 # 2 columns (약 32픽셀) 정도 여유를 둠
#             safe_force_w = max(0, latent_overlap_w - erosion_margin)

#             # Mask 생성 (안전한 영역만 1, 경계선 및 나머지는 0)
#             force_mask = torch.zeros((B, 1, m, n), device=last_tokens.device)
#             if safe_force_w > 0:
#                 force_mask[..., :safe_force_w] = 1.0
                
#             print(f"  [Latent Stitching] Overlap: {latent_overlap_w} -> Force: {safe_force_w} (Eroded {erosion_margin})")

#         # --- B. Inference Input Preparation ---
#         if args.cfg != 1.0:
#             input_ids = class_info['input_ids']
#             attention_mask = class_info['attention_mask']
#             if force_tokens is not None:
#                 force_tokens = force_tokens.repeat(2, 1, 1, 1)
#                 force_mask = force_mask.repeat(2, 1, 1, 1)
#         else:
#             input_ids = class_info['input_ids'][:1]
#             attention_mask = class_info['attention_mask'][:1]

#         # --- C. Sample ---
#         # 수정된 harmon_p는 (pred, tokens)를 반환함
#         generated_img_tensor, generated_tokens = model.sample(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             force_tokens=force_tokens,
#             force_mask=force_mask,
#             image_shape=(args.image_size//16, args.image_size//16),
#             num_iter=64,
#             cfg=args.cfg
#         )

#         # --- D. Save ---
#         last_tokens = generated_tokens # 다음 스텝을 위해 토큰 저장

#         img_np = generated_img_tensor[0].permute(1, 2, 0).float().cpu().numpy()
#         img_np = np.clip((img_np + 1) / 2 * 255, 0, 255).astype(np.uint8)
        
#         Image.fromarray(img_np).save(os.path.join(output_dir, f"view_{i}_latent.jpg"))
#         generated_images.append(img_np)

#     # 5. Final Stitching (Simple Visualization)
#     # 겹치는 부분 제외하고 붙이기
#     final_canvas = [generated_images[0]]
#     overlap_pixel_w = int(args.image_size * args.overlap_ratio)
    
#     for img in generated_images[1:]:
#         non_overlap_part = img[:, overlap_pixel_w:, :]
#         final_canvas.append(non_overlap_part)
    
#     panorama = np.concatenate(final_canvas, axis=1)
#     Image.fromarray(panorama).save(args.output)
#     print(f"Saved to {args.output}")

import torch
import torch.nn.functional as F
from PIL import Image
from mmengine.config import Config
import argparse
import numpy as np
import os
from src.builder import BUILDER

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path.')
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--output", type=str, default='outputs/test_inpainting.jpg')
    args = parser.parse_args()

    # 1. 모델 로드
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).eval().cuda().to(torch.bfloat16)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint, strict=False)
    
    # 2. 텍스트 조건 준비
    prompt = "Generate an image: a panoramic view of a living room"
    class_info = model.prepare_text_conditions(prompt, 'Generate an image.')

    # [수정] 출력 디렉토리 확인 및 생성 (가장 먼저 수행)
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    print("\n>>> Step 1: Reference Image 생성 (Ground Truth)")
    # 아무런 제약 없이 온전한 이미지 하나를 생성합니다.
    with torch.no_grad():
        input_ids = class_info['input_ids']
        attention_mask = class_info['attention_mask']
        if args.cfg != 1.0:
             input_ids = torch.cat([input_ids, input_ids], dim=0) # Cond/Uncond
             attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
        
        # force 없이 순수 생성
        # 주의: harmon_p_temp.py가 (pred, tokens)를 리턴하도록 수정되어 있어야 함.
        # 만약 수정 전이라면 ref_img만 리턴됨. 에러 방지를 위해 체크.
        output = model.sample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_shape=(args.image_size//16, args.image_size//16),
            num_iter=64,
            cfg=args.cfg
        )
        
        if isinstance(output, tuple):
            ref_img, ref_tokens = output
        else:
            ref_img = output
            # 토큰을 못 받았으면 테스트 불가 -> 강제로 추출 시도하거나 에러 처리
            # 여기서는 harmon_p_temp.py가 수정되었다고 가정하고 진행
            raise RuntimeError("harmon_p_temp.py의 sample 함수가 (image, token)을 반환하도록 수정되지 않았습니다.")

    # Reference 저장 (bfloat16 -> float32 캐스팅 추가)
    ref_save_path = os.path.join(output_dir, "step1_reference.jpg")
    Image.fromarray(((ref_img[0].permute(1, 2, 0).float().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)).save(ref_save_path)
    print(f"Saved {ref_save_path}")

    print("\n>>> Step 2: Inpainting Test (Left Half Fixed)")
    
    # [추가된 부분] Latent Scale Check & Fix
    # -------------------------------------------------------------------------
    # 토큰의 값 범위(표준편차)가 너무 크면 이미지가 깨집니다(Deep Fried 현상).
    # 따라서 Std가 비정상적으로 높다면 강제로 줄여주는(Normalize) 로직입니다.
    # token_std = ref_tokens.std().item()
    # print(f"DEBUG: Reference Tokens - Std: {token_std:.4f}")

    # if token_std > 2.0:
    #     print("⚠ WARNING: Latent scale is too high! Normalizing...")
    #     # Harmon/SD VAE의 Scale Factor (0.18215)를 곱해서 원래 범위로 되돌립니다.
    #     vae_scale_factor = 0.18215 
    #     ref_tokens = ref_tokens * vae_scale_factor
    #     print(f"FIXED: New Std: {ref_tokens.std().item():.4f}")
    # # -------------------------------------------------------------------------

    # 방금 만든 토큰(ref_tokens)의 왼쪽 절반을 '강제(Force)'하고, 오른쪽 절반을 다시 그리게 시킵니다.
    B, m, n, d = ref_tokens.shape
    half_w = n // 2
    
    # 왼쪽 절반 토큰 복사
    force_tokens = torch.zeros_like(ref_tokens)
    force_tokens[:, :, :half_w, :] = ref_tokens[:, :, :half_w, :]
    
    # 마스크 설정 (왼쪽: 1=Force, 오른쪽: 0=Generate)
    force_mask = torch.zeros((B, 1, m, n), device=ref_tokens.device)
    force_mask[..., :half_w] = 1.0
    
    # Inference
    with torch.no_grad():
        # CFG용 반복
        force_tokens_in = force_tokens.repeat(2, 1, 1, 1) if args.cfg != 1.0 else force_tokens
        force_mask_in = force_mask.repeat(2, 1, 1, 1) if args.cfg != 1.0 else force_mask
        
        output_inpainted = model.sample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            force_tokens=force_tokens_in,
            force_mask=force_mask_in,
            image_shape=(args.image_size//16, args.image_size//16),
            num_iter=64,
            cfg=args.cfg
        )
        
        if isinstance(output_inpainted, tuple):
            inpainted_img, _ = output_inpainted
        else:
            inpainted_img = output_inpainted

    # Inpainting 결과 저장 (bfloat16 -> float32 캐스팅 추가)
    inpaint_save_path = os.path.join(output_dir, "step2_inpainting.jpg")
    Image.fromarray(((inpainted_img[0].permute(1, 2, 0).float().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)).save(inpaint_save_path)
    print(f"Saved {inpaint_save_path}")