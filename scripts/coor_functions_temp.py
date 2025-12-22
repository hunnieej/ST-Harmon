import math
import torch

def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2.0 * math.pi) - math.pi

def _assign_unique_int_indices(i_float: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """
    Greedy unique integer assignment for a 1D set of targets.

    i_float: (K,) float target positions
    returns: (K,) long assigned unique ints in [lo, hi]
    """
    K = int(i_float.numel())
    order = torch.argsort(i_float)
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
            raise ValueError("[assign_unique] No available integer slot. Increase tile_m/tile_n or reduce owned targets.")

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
    fov_y_deg: float,
    device: torch.device,
):
    """
    Inverse-writer cylindrical yaw mapping with vertical perspective (FoV_y), R=1, H auto.

    Global grid:
      - x -> theta_x
      - y -> y_world in [-H/2, H/2]

    Tiles (yaw-only, perspective):
      alpha(u) = atan(u * tan(FoVx/2))
      y_world(u,v) = R * (v * tan(FoVy/2)) / sqrt(1 + (u*tan(FoVx/2))^2)

    SSOT writer:
      - owner_x partitions global x to tiles
      - for each owned x, assign unique i (injective)
      - for each owned x, assign unique j for all global y into [0..tile_m-1]
        (requires tile_m > m in practice to avoid wrap/permutation)
    """
    T = int(num_tiles)
    if T < 2:
        raise ValueError("--num_tiles must be >= 2")
    if tile_n <= 0 or tile_m <= 0:
        raise ValueError("--tile_m and --tile_n must be > 0")

    fovx = math.radians(float(fov_x_deg))
    fovy = math.radians(float(fov_y_deg))
    if not (0.0 < fovx < math.pi):
        raise ValueError("--fov_x_deg must be in (0, 180)")
    if not (0.0 < fovy < math.pi):
        raise ValueError("--fov_y_deg must be in (0, 180)")

    # coverage in x for nearest-center ownership
    if (fovx / 2.0) + 1e-8 < (math.pi / T):
        raise ValueError(
            f"[cyl-yaw] FoV_x too small: need FoV_x/2 >= pi/T. "
            f"Got FoV_x/2={fovx/2:.4f}, pi/T={math.pi/T:.4f}."
        )

    # R=1, H auto (worst-case owned delta_max = pi/T)
    R = 1.0
    delta_max = math.pi / T
    H = 2.0 * R * math.tan(fovy / 2.0) * math.cos(delta_max)

    # IMPORTANT GUARD:
    # center columns only use |v|<=cos(delta_max). If tile_m==m, unique-j assignment tends to wrap.
    # Give explicit failure instead of silent wrap.
    min_required_tile_m = int(math.ceil(m / max(1e-6, math.cos(delta_max))))
    if tile_m < min_required_tile_m:
        raise ValueError(
            f"[cyl-yaw] tile_m too small for vertical perspective SSOT without wrap.\n"
            f"  m={m}, num_tiles={T}, cos(pi/T)={math.cos(delta_max):.4f}\n"
            f"  Need tile_m >= ceil(m / cos(pi/T)) = {min_required_tile_m}, but got tile_m={tile_m}.\n"
            f"  Fix: increase tile_m (recommended) or drop vertical perspective in writer."
        )

    theta0 = torch.linspace(0.0, 2.0 * math.pi, steps=T + 1, device=device)[:-1]  # (T,)

    x = torch.arange(n, device=device).float()
    theta_x = 2.0 * math.pi * (x + 0.5) / float(n)  # (n,)

    dist = torch.abs(wrap_to_pi(theta_x.view(1, n) - theta0.view(T, 1)))  # (T,n)
    owner_x = torch.argmin(dist, dim=0).long()  # (n,)

    # forward context grid on tile (tile_m x tile_n)
    i = torch.arange(tile_n, device=device).float()
    j = torch.arange(tile_m, device=device).float()
    u = 2.0 * (i + 0.5) / float(tile_n) - 1.0  # (tile_n,)
    v = 2.0 * (j + 0.5) / float(tile_m) - 1.0  # (tile_m,)

    alpha = torch.atan(u * math.tan(fovx / 2.0))  # (tile_n,)
    dx_u = u * math.tan(fovx / 2.0)               # (tile_n,)
    s_u = torch.sqrt(1.0 + dx_u * dx_u).clamp_min(1e-8)  # (tile_n,)

    dy_v = v * math.tan(fovy / 2.0)  # (tile_m,)
    y_world_fwd = (R * dy_v.view(tile_m, 1)) / s_u.view(1, tile_n)  # (tile_m,tile_n)

    y_float_fwd = ((y_world_fwd + (H / 2.0)) / H) * float(m)
    y_idx_fwd = torch.clamp(torch.round(y_float_fwd - 0.5).long(), 0, m - 1)

    # global y_world centers
    y_idx = torch.arange(m, device=device).float()
    y_world_centers = ((y_idx + 0.5) / float(m) - 0.5) * H  # (m,)

    ys_long = torch.arange(m, device=device, dtype=torch.long).view(m, 1)  # (m, This is global y)

    read_lin_list = []
    writer_lin_list = []
    writer_pix_list = []

    for t in range(T):
        xs_owned = (owner_x == t).nonzero(as_tuple=True)[0].long()  # (Kx,)
        Kx = int(xs_owned.numel())
        if Kx == 0:
            raise ValueError(f"[cyl-yaw] tile {t} owns zero columns. Check n/num_tiles.")

        delta = wrap_to_pi(theta_x.index_select(0, xs_owned) - theta0[t])  # (Kx,)
        if torch.max(torch.abs(delta)).item() > (fovx / 2.0 + 1e-6):
            raise ValueError(
                f"[cyl-yaw] Owner tile {t} has a column outside FoV_x. "
                f"max|delta|={torch.max(torch.abs(delta)).item():.6f}, FoV_x/2={fovx/2:.6f}."
            )

        # inverse x
        u_x = torch.tan(delta) / math.tan(fovx / 2.0)  # (Kx,)
        u_x = torch.clamp(u_x, -0.999, 0.999)
        i_float = (u_x + 1.0) * 0.5 * float(tile_n) - 0.5
        i_assigned = _assign_unique_int_indices(i_float, lo=0, hi=tile_n - 1)  # (Kx,)

        # inverse y (per owned x column)
        dx = u_x * math.tan(fovx / 2.0)                           # (Kx,)
        s = torch.sqrt(1.0 + dx * dx).clamp_min(1e-8)             # (Kx,)

        v_yx = (y_world_centers.view(m, 1) * s.view(1, Kx)) / (R * math.tan(fovy / 2.0))  # (m,Kx)
        v_yx = torch.clamp(v_yx, -0.999, 0.999)

        j_float_yx = (v_yx + 1.0) * 0.5 * float(tile_m) - 0.5  # (m,Kx)

        j_assigned_yx = torch.empty((m, Kx), device=device, dtype=torch.long)
        for k in range(Kx):
            j_assigned_yx[:, k] = _assign_unique_int_indices(j_float_yx[:, k], lo=0, hi=tile_m - 1)

        # forward x mapping for context
        theta_cols = torch.remainder(theta0[t] + alpha, 2.0 * math.pi)  # (tile_n,)
        x_float_fwd = theta_cols / (2.0 * math.pi) * float(n)
        x_idx_fwd = torch.remainder(torch.round(x_float_fwd - 0.5).long(), n)  # (tile_n,)

        lin_read = (y_idx_fwd * n + x_idx_fwd.view(1, tile_n)).reshape(-1).long()  # (tile_m*tile_n,)

        # writer
        lin_writer = (ys_long * n + xs_owned.view(1, Kx)).reshape(-1).long()  # (m*Kx,)
        pix_writer = (j_assigned_yx * tile_n + i_assigned.view(1, Kx)).reshape(-1).long()  # (m*Kx,)

        lin_read.index_copy_(0, pix_writer, lin_writer)

        read_lin_list.append(lin_read)
        writer_lin_list.append(lin_writer)
        writer_pix_list.append(pix_writer)

    # sanity: each global cell written exactly once
    all_lin = torch.cat(writer_lin_list, dim=0)
    if int(all_lin.numel()) != int(m * n):
        raise ValueError(f"[cyl-yaw] writer coverage mismatch: got {all_lin.numel()} vs expected {m*n}")
    uniq = torch.unique(all_lin)
    if int(uniq.numel()) != int(m * n):
        raise ValueError("[cyl-yaw] writer_lin has duplicates.")

    return theta0, owner_x, H, read_lin_list, writer_lin_list, writer_pix_list
