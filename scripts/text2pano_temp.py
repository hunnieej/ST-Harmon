import torch
import torch.nn.functional as F
from PIL import Image
from mmengine.config import Config
import argparse
import numpy as np
from tqdm import tqdm
from einops import rearrange
import os
from src.builder import BUILDER
from cylindrical_functions_temp import get_patch_coordinates

def update_global_canvas(canvas, patch, yaw_deg, fov_deg):
    """
    생성된 Patch를 Global Canvas의 해당 위치에 붙여넣는 함수.
    """
    B, C, H_g, W_g = canvas.shape
    _, _, H_p, W_p = patch.shape
    
    # Global Map에서의 중심 X 좌표 계산 ( [-180, 180] -> [0, W_g] )
    normalized_x = (yaw_deg + 180) / 360.0 
    center_x = int(normalized_x * W_g)
    
    # Patch의 폭이 Global Map에서 차지하는 비율 계산
    patch_width_on_global = int(W_g * (fov_deg / 360.0))
    
    start_x = center_x - patch_width_on_global // 2
    end_x = start_x + patch_width_on_global
    
    # Patch를 Global Grid 크기에 맞게 리사이징 (Interpolation)
    patch_resized = F.interpolate(patch, size=(H_g, patch_width_on_global), mode='bicubic')
    
    # Canvas 범위 처리 (Wrap-around 고려)
    if start_x < 0:
        overflow = -start_x
        canvas[..., -overflow:] = patch_resized[..., :overflow]
        canvas[..., :end_x] = patch_resized[..., overflow:]
    elif end_x > W_g:
        overflow = end_x - W_g
        canvas[..., start_x:] = patch_resized[..., :-overflow]
        canvas[..., :overflow] = patch_resized[..., -overflow:]
    else:
        canvas[..., start_x:end_x] = patch_resized
        
    return canvas

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path.')
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--prompt", type=str, default='a panoramic view of a living room')
    parser.add_argument("--num_tiles", type=int, default=8, help="Number of patches to stitch")
    parser.add_argument("--fov", type=float, default=60.0, help="FOV of each patch")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--output", type=str, default='outputs/panorama.jpg')
    args = parser.parse_args()

    # 1. Load Model
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).eval().cuda().to(torch.bfloat16) 
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint, strict=False)

    # 2. Setup Global Canvas
    H_g = args.image_size
    W_g = int(args.image_size * (360 / args.fov)) 
    global_canvas = torch.zeros(1, 3, H_g, W_g).cuda().to(model.dtype)
    
    # 3. Text Embeddings
    args.prompt = f"Generate an image: {args.prompt}"
    print(f"Prompt: {args.prompt}")
    class_info = model.prepare_text_conditions(args.prompt, 'Generate an image.')
    
    # 4. Generation Loop
    yaw_angles = np.linspace(-180, 180, args.num_tiles, endpoint=False)
    
    # 결과 폴더 생성
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for i, yaw in enumerate(tqdm(yaw_angles, desc="Generating Panorama")):
        
        # --- A. Overlap Sampling (Canvas Rolling 방식) ---
        
        # 1. Canvas Rolling: 현재 보는 각도(yaw)가 캔버스의 정중앙(0도)에 오도록 캔버스를 굴립니다.
        # 이렇게 하면 grid_sample이 캔버스의 양 끝단을 넘어갈 일이 사라집니다.
        shift_amount = int(-yaw / 360.0 * W_g)
        canvas_shifted = torch.roll(global_canvas, shifts=shift_amount, dims=-1)
        
        # 2. Grid 생성: 캔버스를 중앙으로 옮겼으니, 카메라는 정면(0도)을 보면 됩니다.
        grid_centered = get_patch_coordinates(H=args.image_size, W=args.image_size, 
                                              fov_deg=args.fov, yaw_deg=0, device='cuda')
        
        # 3. Sampling
        hint_patch = F.grid_sample(canvas_shifted, grid_centered.to(model.dtype), align_corners=True)
        
        # --- 디버깅: 힌트 패치 저장 (View 1부터 확인 가능) ---
        if i > 0:
            debug_path = os.path.join(output_dir, f"debug_hint_view_{i}.jpg")
            debug_img = hint_patch[0].permute(1, 2, 0).float().cpu().numpy()
            debug_img = np.clip((debug_img + 1) / 2 * 255, 0, 255).astype(np.uint8)
            Image.fromarray(debug_img).save(debug_path)
        # ----------------------------------------------------

        # Overlap 감지
        # (힌트 패치에 내용이 있으면 1, 없으면 0)
        is_overlap = (hint_patch.abs().sum(dim=1, keepdim=True) > 0.1).float()
        overlap_ratio = is_overlap.mean().item()
        
        # --- B. Encode Hints ---
        force_tokens = None
        force_mask = None
        
        # [중요] View 0은 overlap_ratio가 0이므로 실행되지 않음 -> force_tokens = None (정상)
        if i > 0 and overlap_ratio > 0.05:
            # print(f"  [View {i}] Overlap Ratio: {overlap_ratio:.2f}")
            with torch.no_grad():
                z = model.encode(hint_patch) 
                force_tokens = z 
                m, n = z.shape[1], z.shape[2]
                force_mask = F.interpolate(is_overlap, size=(m, n), mode='nearest')

        # --- C. Harmon Inference ---
        if args.cfg != 1.0:
            # 1. Text Inputs: 자르지 않고 전체 사용 (Cond + Uncond)
            input_ids = class_info['input_ids']          # Shape: (2, L)
            attention_mask = class_info['attention_mask'] # Shape: (2, L)
            
            # 2. Force Inputs: 배치 크기(2)에 맞춰 복제
            if force_tokens is not None:
                # force_tokens: (1, m, n, d) -> (2, m, n, d)
                force_tokens = force_tokens.repeat(2, 1, 1, 1)
                # force_mask: (1, 1, m, n) -> (2, 1, m, n)
                force_mask = force_mask.repeat(2, 1, 1, 1)
        else:
            # CFG를 안 쓰는 경우 (1개만 사용)
            input_ids = class_info['input_ids'][:1]
            attention_mask = class_info['attention_mask'][:1]
        
        generated_tokens = model.sample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            force_tokens=force_tokens, # View 0에서는 None이 들어감
            force_mask=force_mask,
            image_shape=(args.image_size//16, args.image_size//16),
            num_iter=64,
            cfg=args.cfg,
        )
        
        # --- D. Update Global Canvas ---
        patch_image = generated_tokens 
        global_canvas = update_global_canvas(global_canvas, patch_image, yaw, args.fov)

    # 5. Save Final Panorama
    final_img = global_canvas[0].permute(1, 2, 0).float().cpu().numpy()
    final_img = np.clip((final_img + 1) / 2 * 255, 0, 255).astype(np.uint8)
    Image.fromarray(final_img).save(args.output)
    print(f"Panorama saved to {args.output}")