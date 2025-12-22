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
