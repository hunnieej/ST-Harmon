'''
GOAL : 실린더 좌표계 변환
1) R=1, H 자동
2) Functions
    - get_tangent_grid(fov_deg, H, W)
    - xyz2cyl(x, y, z)
    - sample_from_global 

'''
import torch
import numpy as np
import torch.nn.functional as F

def get_rotation_matrix(yaw_deg, pitch_deg=0, roll_deg=0, device='cpu'):
    """
    카메라의 회전 각도(Euler Angle)를 3x3 회전 행렬로 변환합니다.
    현재는 Yaw(Y축 회전)만 주로 사용합니다.
    """
    # Convert to radians
    yaw = np.deg2rad(yaw_deg)
    # pitch = np.deg2rad(pitch_deg) # 추후 확장 가능
    # roll = np.deg2rad(roll_deg)   # 추후 확장 가능

    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)

    # Rotation Matrix around Y-axis (Vertical Axis)
    # [ cos  0  sin ]
    # [  0   1   0  ]
    # [ -sin 0  cos ]
    rot_mat = torch.tensor([
        [cos_y, 0, sin_y],
        [0,     1, 0    ],
        [-sin_y, 0, cos_y]
    ], device=device, dtype=torch.float32)

    return rot_mat

def create_meshgrid(height, width, fov_deg, device='cpu'):
    """
    1. 2D 이미지 그리드 (u, v) 생성 [-1, 1]
    2. 카메라 내부 파라미터(Intrinsics)를 적용해 Local 3D Ray (x', y', z')로 변환
    """
    # 1. Normalized Device Coordinates (NDC): [-1, 1]
    # grid_y: -1 (Top) -> 1 (Bottom)
    # grid_x: -1 (Left) -> 1 (Right)
    y_range = torch.linspace(-1, 1, height, device=device)
    x_range = torch.linspace(-1, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

    # 2. Apply FOV (Intrinsic)
    # z' = 1 (Forward direction)
    fov_rad = np.deg2rad(fov_deg)
    tan_half_fov = np.tan(fov_rad / 2.0)

    # Aspect Ratio 반영 (assuming square pixels)
    aspect_ratio = height / width
    
    x_local = grid_x * tan_half_fov
    y_local = grid_y * tan_half_fov * aspect_ratio
    z_local = torch.ones_like(x_local)

    # Stack: (H, W, 3)
    rays_local = torch.stack([x_local, y_local, z_local], dim=-1)
    
    return rays_local, tan_half_fov, aspect_ratio

def cart2cyl(x, y, z):
    """
    Global 3D 좌표 (x, y, z) -> 원통 좌표 (theta, h)
    theta: [-pi, pi] (Longitude)
    h: Height value (Projection on cylinder surface)
    """
    # 1. Longitude (Theta)
    # atan2(x, z)는 (-pi, pi) 반환. 
    # 좌표계 정의에 따라 부호가 달라질 수 있으나, 일반적으로 z-forward, x-right 기준.
    theta = torch.atan2(x, z) 

    # 2. Height (h)
    # 원통 투영: 중심에서의 거리로 y를 스케일링
    # r = sqrt(x^2 + z^2)
    dist_xz = torch.sqrt(x**2 + z**2 + 1e-8)
    h = y / dist_xz

    return theta, h

def get_patch_coordinates(H, W, fov_deg, yaw_deg, device='cuda'):
    """
    위 함수들을 조합하여 grid_sample에 사용할 Sampling Grid를 생성하는 Wrapper 함수.
    
    Returns:
        grid (Tensor): (1, H, W, 2) 크기, 값 범위 [-1, 1]
                       (x=theta_norm, y=h_norm) 순서
    """
    # 1. Create Local Rays
    rays_local, tan_half_fov, aspect_ratio = create_meshgrid(H, W, fov_deg, device=device)
    
    # 2. Get Rotation Matrix & Rotate to Global
    rot_mat = get_rotation_matrix(yaw_deg, device=device)
    
    # (H, W, 3) @ (3, 3).T -> (H, W, 3)
    rays_global = torch.matmul(rays_local, rot_mat.T)
    
    x_global = rays_global[..., 0]
    y_global = rays_global[..., 1]
    z_global = rays_global[..., 2]

    # 3. Cartesian to Cylindrical
    theta, h = cart2cyl(x_global, y_global, z_global)

    # 4. Normalize for grid_sample [-1, 1]
    
    # Theta: [-pi, pi] -> [-1, 1]
    theta_norm = theta / np.pi
    
    # Height: 
    # 패치의 수직 시야각 경계(image top/bottom)가 grid의 -1/1에 해당하도록 정규화
    # h_max = y_local_max / 1 = tan(fov/2) * aspect_ratio
    h_max = tan_half_fov * aspect_ratio
    h_norm = h / h_max

    # 5. Stack for grid_sample (x, y order)
    # Shape: (H, W, 2)
    grid = torch.stack([theta_norm, h_norm], dim=-1)
    
    # Add Batch Dimension: (1, H, W, 2)
    return grid.unsqueeze(0)