"""cylindrical_functions_temp.py

Unified projection utilities for panorama / tangent-view generation.

Design goals
------------
1) Keep everything in ONE module, while making responsibilities explicit.
2) Provide BOTH:
   (a) continuous sampling grids (for grid_sample / warping)
   (b) discrete token indexing maps (read_lin / writer_lin / writer_pix) for SSOT scheduling
3) Make it easy to extend to spherical projection later.

Key entrypoint
--------------
    build_projection_ssot_maps(projection_type=..., ...)

Currently supported:
  - projection_type="cyl_yaw" : cylindrical surface, yaw-only camera tiles (production)
  - projection_type="sph_demo": spherical surface, simple yaw-only tiles (demo / placeholder)

Notes
-----
* The SSOT 'unique writer' logic is centralized (common) and projection-specific code only
  supplies the float targets needed to compute writer pixel indices.
* This module intentionally does NOT assume any package layout. If you keep it under
  scripts/, it can be imported as scripts.cylindrical_functions_temp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch


# ============================================================
# Small math helpers
# ============================================================

def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-pi, pi]."""
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def deg2rad(x: float) -> float:
    return float(x) * math.pi / 180.0


def _assign_unique_int_indices(i_float: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Greedy unique integer assignment for a 1D set of targets.

    Parameters
    ----------
    i_float: (K,) float targets (preferred positions)
    lo, hi: inclusive integer range

    Returns
    -------
    assigned: (K,) long, unique ints in [lo, hi]

    Rationale
    ---------
    SSOT requires an injective mapping from global cells -> tile pixels.
    This helper turns float targets into unique integer slots.
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
            raise ValueError(
                "[assign_unique] No available integer slot. "
                "Increase tile_m/tile_n or reduce owned targets."
            )

    return assigned


# ============================================================
# Continuous projection (for grid_sample / warping)
# ============================================================

def get_rotation_matrix(
    yaw_deg: float,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Euler angles -> 3x3 rotation matrix.

    Convention: yaw (Y-axis), pitch (X-axis), roll (Z-axis).
    This matches a common camera convention (z-forward, x-right, y-up) when applied
    appropriately, but you may want to verify sign conventions in your pipeline.
    """
    device = torch.device(device)

    yaw = torch.tensor(deg2rad(yaw_deg), device=device, dtype=dtype)
    pitch = torch.tensor(deg2rad(pitch_deg), device=device, dtype=dtype)
    roll = torch.tensor(deg2rad(roll_deg), device=device, dtype=dtype)

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    # Yaw about Y
    Ry = torch.stack(
        [
            torch.stack([cy, torch.zeros_like(cy), sy]),
            torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)]),
            torch.stack([-sy, torch.zeros_like(cy), cy]),
        ],
        dim=0,
    )
    # Pitch about X
    Rx = torch.stack(
        [
            torch.stack([torch.ones_like(cp), torch.zeros_like(cp), torch.zeros_like(cp)]),
            torch.stack([torch.zeros_like(cp), cp, -sp]),
            torch.stack([torch.zeros_like(cp), sp, cp]),
        ],
        dim=0,
    )
    # Roll about Z
    Rz = torch.stack(
        [
            torch.stack([cr, -sr, torch.zeros_like(cr)]),
            torch.stack([sr, cr, torch.zeros_like(cr)]),
            torch.stack([torch.zeros_like(cr), torch.zeros_like(cr), torch.ones_like(cr)]),
        ],
        dim=0,
    )

    # Apply roll -> pitch -> yaw (intrinsic). Adjust if needed.
    return Ry @ Rx @ Rz


def create_meshgrid(
    height: int,
    width: int,
    *,
    fov_x_deg: float,
    fov_y_deg: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create local ray directions (H, W, 3) for a pinhole camera.

    Returns rays pointing forward (z=1) with x/y scaled by tan(FoV/2).
    """
    device = torch.device(device)
    y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    tan_half_fovx = math.tan(deg2rad(fov_x_deg) / 2.0)
    tan_half_fovy = math.tan(deg2rad(fov_y_deg) / 2.0)

    x_local = grid_x * tan_half_fovx
    y_local = grid_y * tan_half_fovy
    z_local = torch.ones_like(x_local)
    return torch.stack([x_local, y_local, z_local], dim=-1)


def cart2cyl(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cartesian -> cylindrical surface coords.

    theta in (-pi, pi], h = y / sqrt(x^2+z^2)
    """
    theta = torch.atan2(x, z)
    dist_xz = torch.sqrt(x * x + z * z + 1e-8)
    h = y / dist_xz
    return theta, h


def cart2sph(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cartesian -> spherical surface coords (theta, phi).

    theta: longitude (-pi, pi]
    phi  : latitude  [-pi/2, pi/2]
    """
    theta = torch.atan2(x, z)
    r = torch.sqrt(x * x + y * y + z * z + 1e-8)
    phi = torch.asin(torch.clamp(y / r, -1.0, 1.0))
    return theta, phi


def get_patch_coordinates_cyl(
    *,
    H: int,
    W: int,
    fov_x_deg: float,
    fov_y_deg: float,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """grid_sample grid for sampling a cylindrical panorama in (theta, h) parameterization.

    Output: (1, H, W, 2) with range roughly [-1,1] for both axes.
    """
    rays_local = create_meshgrid(H, W, fov_x_deg=fov_x_deg, fov_y_deg=fov_y_deg, device=device)
    R = get_rotation_matrix(yaw_deg, pitch_deg, roll_deg, device=device)
    rays_global = rays_local @ R.T
    theta, h = cart2cyl(rays_global[..., 0], rays_global[..., 1], rays_global[..., 2])

    # Normalize theta (-pi,pi] -> [-1,1]
    theta_norm = theta / math.pi
    # Normalize h by tan(fov_y/2)
    h_max = math.tan(deg2rad(fov_y_deg) / 2.0)
    h_norm = h / max(h_max, 1e-8)
    grid = torch.stack([theta_norm, h_norm], dim=-1)
    return grid.unsqueeze(0)


def get_patch_coordinates_sph(
    *,
    H: int,
    W: int,
    fov_x_deg: float,
    fov_y_deg: float,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """grid_sample grid for sampling an equirectangular sphere in (theta, phi).

    Output: (1, H, W, 2) where
      x = theta_norm in [-1,1]
      y = phi_norm   in [-1,1]
    """
    rays_local = create_meshgrid(H, W, fov_x_deg=fov_x_deg, fov_y_deg=fov_y_deg, device=device)
    R = get_rotation_matrix(yaw_deg, pitch_deg, roll_deg, device=device)
    rays_global = rays_local @ R.T
    theta, phi = cart2sph(rays_global[..., 0], rays_global[..., 1], rays_global[..., 2])

    theta_norm = theta / math.pi
    phi_norm = phi / (math.pi / 2.0)
    grid = torch.stack([theta_norm, phi_norm], dim=-1)
    return grid.unsqueeze(0)


# ============================================================
# Discrete token indexing (SSOT) : unified entrypoint
# ============================================================

ProjectionType = Literal["cyl_yaw", "sph_demo"]


@dataclass(frozen=True)
class ProjectionSSoTMaps:
    """Precomputed maps for tangent-tile inference with global SSOT schedule."""

    projection_type: str
    m: int
    n: int
    tile_m: int
    tile_n: int
    num_tiles: int
    fov_x_deg: float
    fov_y_deg: float

    read_lin_list: List[torch.Tensor]
    writer_lin_list: List[torch.Tensor]
    writer_pix_list: List[torch.Tensor]

    meta: Dict[str, torch.Tensor | float | int]


class _BaseAdapter:
    """Projection-specific provider for forward maps and inverse float targets."""

    def __init__(
        self,
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
        self.m = int(m)
        self.n = int(n)
        self.tile_m = int(tile_m)
        self.tile_n = int(tile_n)
        self.T = int(num_tiles)
        self.fovx = deg2rad(float(fov_x_deg))
        self.fovy = deg2rad(float(fov_y_deg))
        self.device = device

        if self.T < 2:
            raise ValueError("--num_tiles must be >= 2")
        if self.tile_n <= 0 or self.tile_m <= 0:
            raise ValueError("--tile_m and --tile_n must be > 0")
        if not (0.0 < self.fovx < math.pi):
            raise ValueError("--fov_x_deg must be in (0, 180)")
        if not (0.0 < self.fovy < math.pi):
            raise ValueError("--fov_y_deg must be in (0, 180)")

    def prepare(self) -> Dict[str, object]:
        """Return all per-tile ingredients for the common SSOT builder.

        Required keys:
          - read_lin_list: List[LongTensor] each (tile_m*tile_n,)
          - lin_writer_list: List[LongTensor] each (m*Kx,)
          - i_float_list: List[FloatTensor] each (Kx,)
          - j_float_yx_list: List[FloatTensor] each (m,Kx)
          - meta: Dict
        """
        raise NotImplementedError


class _CylYawAdapter(_BaseAdapter):
    """Production adapter: matches the original precompute_cyl_yaw_inverse_writer behavior."""

    def prepare(self) -> Dict[str, object]:
        m, n, tile_m, tile_n, T = self.m, self.n, self.tile_m, self.tile_n, self.T
        fovx, fovy = self.fovx, self.fovy
        device = self.device

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
        u = 2.0 * (i + 0.5) / float(tile_n) - 1.0
        v = 2.0 * (j + 0.5) / float(tile_m) - 1.0

        alpha = torch.atan(u * math.tan(fovx / 2.0))
        dx_u = u * math.tan(fovx / 2.0)
        s_u = torch.sqrt(1.0 + dx_u * dx_u).clamp_min(1e-8)

        dy_v = v * math.tan(fovy / 2.0)
        y_world_fwd = (R * dy_v.view(tile_m, 1)) / s_u.view(1, tile_n)  # (tile_m,tile_n)
        y_float_fwd = ((y_world_fwd + (H / 2.0)) / H) * float(m)
        y_idx_fwd = torch.clamp(torch.round(y_float_fwd - 0.5).long(), 0, m - 1)

        # global y_world centers
        y_idx = torch.arange(m, device=device).float()
        y_world_centers = ((y_idx + 0.5) / float(m) - 0.5) * H  # (m,)

        ys_long = torch.arange(m, device=device, dtype=torch.long).view(m, 1)

        read_lin_list: List[torch.Tensor] = []
        lin_writer_list: List[torch.Tensor] = []
        i_float_list: List[torch.Tensor] = []
        j_float_yx_list: List[torch.Tensor] = []

        for t in range(T):
            xs_owned = (owner_x == t).nonzero(as_tuple=True)[0].long()  # (Kx,)
            Kx = int(xs_owned.numel())
            if Kx == 0:
                raise ValueError(f"[cyl-yaw] tile {t} owns zero columns. Check n/num_tiles.")

            delta = wrap_to_pi(theta_x.index_select(0, xs_owned) - theta0[t])
            if torch.max(torch.abs(delta)).item() > (fovx / 2.0 + 1e-6):
                raise ValueError(
                    f"[cyl-yaw] Owner tile {t} has a column outside FoV_x. "
                    f"max|delta|={torch.max(torch.abs(delta)).item():.6f}, FoV_x/2={fovx/2:.6f}."
                )

            # inverse x targets
            u_x = torch.tan(delta) / math.tan(fovx / 2.0)
            u_x = torch.clamp(u_x, -0.999, 0.999)
            i_float = (u_x + 1.0) * 0.5 * float(tile_n) - 0.5  # (Kx,)

            # inverse y targets (depend on x via perspective)
            dx = u_x * math.tan(fovx / 2.0)
            s = torch.sqrt(1.0 + dx * dx).clamp_min(1e-8)  # (Kx,)
            v_yx = (y_world_centers.view(m, 1) * s.view(1, Kx)) / (R * math.tan(fovy / 2.0))
            v_yx = torch.clamp(v_yx, -0.999, 0.999)
            j_float_yx = (v_yx + 1.0) * 0.5 * float(tile_m) - 0.5  # (m,Kx)

            # forward x mapping for context
            theta_cols = torch.remainder(theta0[t] + alpha, 2.0 * math.pi)
            x_float_fwd = theta_cols / (2.0 * math.pi) * float(n)
            x_idx_fwd = torch.remainder(torch.round(x_float_fwd - 0.5).long(), n)

            lin_read = (y_idx_fwd * n + x_idx_fwd.view(1, tile_n)).reshape(-1).long()
            lin_writer = (ys_long * n + xs_owned.view(1, Kx)).reshape(-1).long()

            read_lin_list.append(lin_read)
            lin_writer_list.append(lin_writer)
            i_float_list.append(i_float)
            j_float_yx_list.append(j_float_yx)

        meta: Dict[str, object] = {
            "theta0": theta0,
            "owner_x": owner_x,
            "H": float(H),
            "delta_max": float(delta_max),
        }

        return {
            "read_lin_list": read_lin_list,
            "lin_writer_list": lin_writer_list,
            "i_float_list": i_float_list,
            "j_float_yx_list": j_float_yx_list,
            "meta": meta,
        }


class _SphDemoAdapter(_BaseAdapter):
    """Demo spherical adapter.

    This is intentionally simple: yaw-only tiles with pitch center fixed at 0.
    Ownership is assigned by nearest yaw-center in theta (like cyl). This is NOT
    ideal near poles and is provided only as a scaffolding for later work.
    """

    def prepare(self) -> Dict[str, object]:
        m, n, tile_m, tile_n, T = self.m, self.n, self.tile_m, self.tile_n, self.T
        fovx, fovy = self.fovx, self.fovy
        device = self.device

        # Basic x-coverage guard similar to cyl.
        if (fovx / 2.0) + 1e-8 < (math.pi / T):
            raise ValueError(
                f"[sph-demo] FoV_x too small: need FoV_x/2 >= pi/T. "
                f"Got FoV_x/2={fovx/2:.4f}, pi/T={math.pi/T:.4f}."
            )

        theta0 = torch.linspace(0.0, 2.0 * math.pi, steps=T + 1, device=device)[:-1]

        # global theta/phi centers
        x = torch.arange(n, device=device).float()
        theta_x = 2.0 * math.pi * (x + 0.5) / float(n)
        y = torch.arange(m, device=device).float()
        phi_y = ( (y + 0.5) / float(m) - 0.5) * math.pi  # [-pi/2, pi/2]

        dist = torch.abs(wrap_to_pi(theta_x.view(1, n) - theta0.view(T, 1)))
        owner_x = torch.argmin(dist, dim=0).long()

        # tile pixel grid
        i = torch.arange(tile_n, device=device).float()
        j = torch.arange(tile_m, device=device).float()
        u = 2.0 * (i + 0.5) / float(tile_n) - 1.0
        v = 2.0 * (j + 0.5) / float(tile_m) - 1.0

        # forward mapping from tile pixels -> (theta,phi)
        alpha = torch.atan(u * math.tan(fovx / 2.0))  # yaw offset
        beta = torch.atan(v * math.tan(fovy / 2.0))   # pitch offset

        # phi depends only on v here (demo). theta depends on tile yaw.
        # Map (theta,phi) to global indices
        phi_norm = (beta + (math.pi / 2.0)) / math.pi  # [0,1]
        y_float_fwd = phi_norm.view(tile_m, 1) * float(m)
        y_idx_fwd = torch.clamp(torch.round(y_float_fwd - 0.5).long(), 0, m - 1)

        ys_long = torch.arange(m, device=device, dtype=torch.long).view(m, 1)

        read_lin_list: List[torch.Tensor] = []
        lin_writer_list: List[torch.Tensor] = []
        i_float_list: List[torch.Tensor] = []
        j_float_yx_list: List[torch.Tensor] = []

        # precompute inverse-y float targets (independent of x in demo)
        v_y = torch.tan(phi_y) / max(1e-8, math.tan(fovy / 2.0))
        v_y = torch.clamp(v_y, -0.999, 0.999)
        j_float_y = (v_y + 1.0) * 0.5 * float(tile_m) - 0.5  # (m,)

        for t in range(T):
            xs_owned = (owner_x == t).nonzero(as_tuple=True)[0].long()
            Kx = int(xs_owned.numel())
            if Kx == 0:
                raise ValueError(f"[sph-demo] tile {t} owns zero columns. Check n/num_tiles.")

            delta = wrap_to_pi(theta_x.index_select(0, xs_owned) - theta0[t])
            u_x = torch.tan(delta) / max(1e-8, math.tan(fovx / 2.0))
            u_x = torch.clamp(u_x, -0.999, 0.999)
            i_float = (u_x + 1.0) * 0.5 * float(tile_n) - 0.5

            # broadcast demo j targets (same for each owned column)
            j_float_yx = j_float_y.view(m, 1).expand(m, Kx).contiguous()

            # forward x mapping
            theta_cols = torch.remainder(theta0[t] + alpha, 2.0 * math.pi)
            x_float_fwd = theta_cols / (2.0 * math.pi) * float(n)
            x_idx_fwd = torch.remainder(torch.round(x_float_fwd - 0.5).long(), n)

            lin_read = (y_idx_fwd * n + x_idx_fwd.view(1, tile_n)).reshape(-1).long()
            lin_writer = (ys_long * n + xs_owned.view(1, Kx)).reshape(-1).long()

            read_lin_list.append(lin_read)
            lin_writer_list.append(lin_writer)
            i_float_list.append(i_float)
            j_float_yx_list.append(j_float_yx)

        meta: Dict[str, object] = {
            "theta0": theta0,
            "owner_x": owner_x,
            "phi_center": 0.0,
        }
        return {
            "read_lin_list": read_lin_list,
            "lin_writer_list": lin_writer_list,
            "i_float_list": i_float_list,
            "j_float_yx_list": j_float_yx_list,
            "meta": meta,
        }


def _build_ssot_from_adapter(
    *,
    m: int,
    n: int,
    tile_m: int,
    tile_n: int,
    read_lin_list: List[torch.Tensor],
    lin_writer_list: List[torch.Tensor],
    i_float_list: List[torch.Tensor],
    j_float_yx_list: List[torch.Tensor],
    validate: bool,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Common SSOT builder.

    For each tile t:
      - assign unique i indices for owned x-columns
      - assign unique j indices per owned column for all y
      - build pix_writer and inject writer cells into read_lin (self-consistency)
    """
    T = len(read_lin_list)
    writer_pix_list: List[torch.Tensor] = []
    writer_lin_list: List[torch.Tensor] = []
    new_read_lin_list: List[torch.Tensor] = []

    for t in range(T):
        lin_read = read_lin_list[t].clone()
        lin_writer = lin_writer_list[t]
        i_float = i_float_list[t]
        j_float_yx = j_float_yx_list[t]

        Kx = int(i_float.numel())
        if Kx <= 0:
            raise ValueError(f"[SSOT] Tile {t} has no owned columns (Kx=0).")

        i_assigned = _assign_unique_int_indices(i_float, lo=0, hi=tile_n - 1)  # (Kx,)

        # Assign j per column (m,Kx)
        j_assigned_yx = torch.empty((m, Kx), device=j_float_yx.device, dtype=torch.long)
        for k in range(Kx):
            j_assigned_yx[:, k] = _assign_unique_int_indices(j_float_yx[:, k], lo=0, hi=tile_m - 1)

        pix_writer = (j_assigned_yx * tile_n + i_assigned.view(1, Kx)).reshape(-1).long()

        # Inject: for those writer pixels, ensure read uses exact writer global indices.
        lin_read.index_copy_(0, pix_writer, lin_writer)

        new_read_lin_list.append(lin_read)
        writer_lin_list.append(lin_writer)
        writer_pix_list.append(pix_writer)

    if validate:
        all_lin = torch.cat(writer_lin_list, dim=0)
        if int(all_lin.numel()) != int(m * n):
            raise ValueError(f"[SSOT] writer coverage mismatch: got {all_lin.numel()} vs expected {m*n}")
        uniq = torch.unique(all_lin)
        if int(uniq.numel()) != int(m * n):
            raise ValueError("[SSOT] writer_lin has duplicates.")

    return new_read_lin_list, writer_lin_list, writer_pix_list


@torch.no_grad()
def build_projection_ssot_maps(
    *,
    projection_type: ProjectionType,
    m: int,
    n: int,
    tile_m: int,
    tile_n: int,
    num_tiles: int,
    fov_x_deg: float,
    fov_y_deg: float,
    device: torch.device,
    validate: bool = True,
) -> ProjectionSSoTMaps:
    """Unified precompute for discrete SSOT maps.

    Parameters mirror the original cylindrical precompute. The return type is unified.
    """
    if projection_type == "cyl_yaw":
        adapter: _BaseAdapter = _CylYawAdapter(
            m=m,
            n=n,
            tile_m=tile_m,
            tile_n=tile_n,
            num_tiles=num_tiles,
            fov_x_deg=fov_x_deg,
            fov_y_deg=fov_y_deg,
            device=device,
        )
    elif projection_type == "sph_demo":
        adapter = _SphDemoAdapter(
            m=m,
            n=n,
            tile_m=tile_m,
            tile_n=tile_n,
            num_tiles=num_tiles,
            fov_x_deg=fov_x_deg,
            fov_y_deg=fov_y_deg,
            device=device,
        )
    else:
        raise ValueError(f"Unknown projection_type={projection_type}")

    prepared = adapter.prepare()
    read_lin_list = prepared["read_lin_list"]  # type: ignore[assignment]
    lin_writer_list = prepared["lin_writer_list"]  # type: ignore[assignment]
    i_float_list = prepared["i_float_list"]  # type: ignore[assignment]
    j_float_yx_list = prepared["j_float_yx_list"]  # type: ignore[assignment]
    meta = prepared.get("meta", {})  # type: ignore[assignment]

    new_read_lin_list, writer_lin_list, writer_pix_list = _build_ssot_from_adapter(
        m=int(m),
        n=int(n),
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        read_lin_list=read_lin_list,
        lin_writer_list=lin_writer_list,
        i_float_list=i_float_list,
        j_float_yx_list=j_float_yx_list,
        validate=bool(validate),
    )

    return ProjectionSSoTMaps(
        projection_type=str(projection_type),
        m=int(m),
        n=int(n),
        tile_m=int(tile_m),
        tile_n=int(tile_n),
        num_tiles=int(num_tiles),
        fov_x_deg=float(fov_x_deg),
        fov_y_deg=float(fov_y_deg),
        read_lin_list=new_read_lin_list,
        writer_lin_list=writer_lin_list,
        writer_pix_list=writer_pix_list,
        meta=meta,  # type: ignore[arg-type]
    )