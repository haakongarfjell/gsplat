import struct
from typing import Optional, Tuple
from typing_extensions import Literal, assert_never

import torch
import torch.nn.functional as F
from torch import Tensor


def _quat_to_rotmat(quats: Tensor) -> Tensor:
    """Convert quaternion to rotation matrix."""
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    R = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return R.reshape(quats.shape[:-1] + (3, 3))


def _quat_scale_to_matrix(
    quats: Tensor,  # [N, 4],
    scales: Tensor,  # [N, 3],
) -> Tensor:
    """Convert quaternion and scale to a 3x3 matrix (R * S)."""
    R = _quat_to_rotmat(quats)  # (..., 3, 3)
    M = R * scales[..., None, :]  # (..., 3, 3)
    return M


def _quat_scale_to_covar_preci(
    quats: Tensor,  # [N, 4],
    scales: Tensor,  # [N, 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.quat_scale_to_covar_preci()`."""
    R = _quat_to_rotmat(quats)  # (..., 3, 3)

    if compute_covar:
        M = R * scales[..., None, :]  # (..., 3, 3)
        covars = torch.bmm(M, M.transpose(-1, -2))  # (..., 3, 3)
        if triu:
            covars = covars.reshape(covars.shape[:-2] + (9,))  # (..., 9)
            covars = (
                covars[..., [0, 1, 2, 4, 5, 8]] + covars[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0  # (..., 6)
    if compute_preci:
        P = R * (1 / scales[..., None, :])  # (..., 3, 3)
        precis = torch.bmm(P, P.transpose(-1, -2))  # (..., 3, 3)
        if triu:
            precis = precis.reshape(precis.shape[:-2] + (9,))
            precis = (
                precis[..., [0, 1, 2, 4, 5, 8]] + precis[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0

    return covars if compute_covar else None, precis if compute_preci else None


def _persp_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of perspective projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [C, N, 3].
        covars: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
        Ks: Camera intrinsics. [C, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [C, N, 2].
        - **cov2d**: Projected covariances. [C, N, 2, 2].
    """
    C, N, _ = means.shape

    tx, ty, tz = torch.unbind(means, dim=-1)  # [C, N]
    tz2 = tz**2  # [C, N]

    fx = Ks[..., 0, 0, None]  # [C, 1]
    fy = Ks[..., 1, 1, None]  # [C, 1]
    cx = Ks[..., 0, 2, None]  # [C, 1]
    cy = Ks[..., 1, 2, None]  # [C, 1]
    tan_fovx = 0.5 * width / fx  # [C, 1]
    tan_fovy = 0.5 * height / fy  # [C, 1]

    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)

    O = torch.zeros((C, N), device=means.device, dtype=means.dtype)
    J = torch.stack(
        [fx / tz, O, -fx * tx / tz2, O, fy / tz, -fy * ty / tz2], dim=-1
    ).reshape(C, N, 2, 3)

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    means2d = torch.einsum("cij,cnj->cni", Ks[:, :2, :3], means)  # [C, N, 2]
    means2d = means2d / tz[..., None]  # [C, N, 2]
    return means2d, cov2d  # [C, N, 2], [C, N, 2, 2]


def _fisheye_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of fisheye projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [C, N, 3].
        covars: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
        Ks: Camera intrinsics. [C, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [C, N, 2].
        - **cov2d**: Projected covariances. [C, N, 2, 2].
    """
    C, N, _ = means.shape

    x, y, z = torch.unbind(means, dim=-1)  # [C, N]

    fx = Ks[..., 0, 0, None]  # [C, 1]
    fy = Ks[..., 1, 1, None]  # [C, 1]
    cx = Ks[..., 0, 2, None]  # [C, 1]
    cy = Ks[..., 1, 2, None]  # [C, 1]

    eps = 0.0000001
    xy_len = (x**2 + y**2) ** 0.5 + eps
    theta = torch.atan2(xy_len, z + eps)
    means2d = torch.stack(
        [
            x * fx * theta / xy_len + cx,
            y * fy * theta / xy_len + cy,
        ],
        dim=-1,
    )

    x2 = x * x + eps
    y2 = y * y
    xy = x * y
    x2y2 = x2 + y2
    x2y2z2_inv = 1.0 / (x2y2 + z * z)
    b = torch.atan2(xy_len, z) / xy_len / x2y2
    a = z * x2y2z2_inv / (x2y2)
    J = torch.stack(
        [
            fx * (x2 * a + y2 * b),
            fx * xy * (a - b),
            -fx * x * x2y2z2_inv,
            fy * xy * (a - b),
            fy * (y2 * a + x2 * b),
            -fy * y * x2y2z2_inv,
        ],
        dim=-1,
    ).reshape(C, N, 2, 3)

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    return means2d, cov2d  # [C, N, 2], [C, N, 2, 2]


def _ortho_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of orthographic projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [C, N, 3].
        covars: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
        Ks: Camera intrinsics. [C, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [C, N, 2].
        - **cov2d**: Projected covariances. [C, N, 2, 2].
    """
    C, N, _ = means.shape

    fx = Ks[..., 0, 0, None]  # [C, 1]
    fy = Ks[..., 1, 1, None]  # [C, 1]

    O = torch.zeros((C, 1), device=means.device, dtype=means.dtype)
    J = torch.stack([fx, O, O, O, fy, O], dim=-1).reshape(C, 1, 2, 3).repeat(1, N, 1, 1)

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    means2d = (
        means[..., :2] * Ks[:, None, [0, 1], [0, 1]] + Ks[:, None, [0, 1], [2, 2]]
    )  # [C, N, 2]
    return means2d, cov2d  # [C, N, 2], [C, N, 2, 2]


def _world_to_cam(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of world to camera transformation on Gaussians.

    Args:
        means: Gaussian means in world coordinate system. [C, N, 3].
        covars: Gaussian covariances in world coordinate system. [C, N, 3, 3].
        viewmats: world to camera transformation matrices. [C, 4, 4].

    Returns:
        A tuple:

        - **means_c**: Gaussian means in camera coordinate system. [C, N, 3].
        - **covars_c**: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
    """
    R = viewmats[:, :3, :3]  # [C, 3, 3]
    t = viewmats[:, :3, 3]  # [C, 3]
    means_c = torch.einsum("cij,nj->cni", R, means) + t[:, None, :]  # (C, N, 3)
    covars_c = torch.einsum("cij,njk,clk->cnil", R, covars, R)  # [C, N, 3, 3]
    return means_c, covars_c


def _fully_fused_projection(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.fully_fused_projection()`

    .. note::

        This is a minimal implementation of fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    means_c, covars_c = _world_to_cam(means, covars, viewmats)

    if camera_model == "ortho":
        means2d, covars2d = _ortho_proj(means_c, covars_c, Ks, width, height)
    elif camera_model == "fisheye":
        means2d, covars2d = _fisheye_proj(means_c, covars_c, Ks, width, height)
    elif camera_model == "pinhole":
        means2d, covars2d = _persp_proj(means_c, covars_c, Ks, width, height)
    else:
        assert_never(camera_model)

    det_orig = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d

    det = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    det = det.clamp(min=1e-10)

    if calc_compensations:
        compensations = torch.sqrt(torch.clamp(det_orig / det, min=0.0))
    else:
        compensations = None

    conics = torch.stack(
        [
            covars2d[..., 1, 1] / det,
            -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
            covars2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # [C, N, 3]

    depths = means_c[..., 2]  # [C, N]

    b = (covars2d[..., 0, 0] + covars2d[..., 1, 1]) / 2  # (...,)
    v1 = b + torch.sqrt(torch.clamp(b**2 - det, min=0.01))  # (...,)
    radius = torch.ceil(3.0 * torch.sqrt(v1))  # (...,)
    # v2 = b - torch.sqrt(torch.clamp(b**2 - det, min=0.01))  # (...,)
    # radius = torch.ceil(3.0 * torch.sqrt(torch.max(v1, v2)))  # (...,)

    valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    inside = (
        (means2d[..., 0] + radius > 0)
        & (means2d[..., 0] - radius < width)
        & (means2d[..., 1] + radius > 0)
        & (means2d[..., 1] - radius < height)
    )
    radius[~inside] = 0.0

    radii = radius.int()
    return radii, means2d, depths, conics, compensations

def _fully_fused_projection2(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Tensor]:
    """PyTorch implementation of `gsplat.cuda._wrapper.fully_fused_projection()` with indices of kept Gaussians.

    .. note::

        This is a minimal implementation of fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    means_c, covars_c = _world_to_cam(means, covars, viewmats)

    if camera_model == "ortho":
        means2d, covars2d = _ortho_proj(means_c, covars_c, Ks, width, height)
    elif camera_model == "fisheye":
        means2d, covars2d = _fisheye_proj(means_c, covars_c, Ks, width, height)
    elif camera_model == "pinhole":
        means2d, covars2d = _persp_proj(means_c, covars_c, Ks, width, height)
    else:
        assert_never(camera_model)

    det_orig = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d

    det = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    det = det.clamp(min=1e-10)

    if calc_compensations:
        compensations = torch.sqrt(torch.clamp(det_orig / det, min=0.0))
    else:
        compensations = None

    conics = torch.stack(
        [
            covars2d[..., 1, 1] / det,
            -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
            covars2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # [C, N, 3]

    depths = means_c[..., 2]  # [C, N]

    b = (covars2d[..., 0, 0] + covars2d[..., 1, 1]) / 2  # (...,)
    v1 = b + torch.sqrt(torch.clamp(b**2 - det, min=0.01))  # (...,)
    radius = torch.ceil(3.0 * torch.sqrt(v1))  # (...,)

    valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    inside = (
        (means2d[..., 0] + radius > 0)
        & (means2d[..., 0] - radius < width)
        & (means2d[..., 1] + radius > 0)
        & (means2d[..., 1] - radius < height)
    )
    radius[~inside] = 0.0

    radii = radius.int()

    # Calculate the mask of valid Gaussians
    final_valid_mask = valid & inside  # [C, N]

    # Extract indices of valid Gaussians
    valid_indices = final_valid_mask.nonzero(as_tuple=False)[:, 1]  # Extract Gaussian indices

    return valid_indices


@torch.no_grad()
def _isect_tiles(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pytorch implementation of `gsplat.cuda._wrapper.isect_tiles()`.

    .. note::

        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    C, N = means2d.shape[:2]
    device = means2d.device

    # compute tiles_per_gauss
    tile_means2d = means2d / tile_size
    tile_radii = radii / tile_size
    tile_mins = torch.floor(tile_means2d - tile_radii[..., None]).int()
    tile_maxs = torch.ceil(tile_means2d + tile_radii[..., None]).int()
    tile_mins[..., 0] = torch.clamp(tile_mins[..., 0], 0, tile_width)
    tile_mins[..., 1] = torch.clamp(tile_mins[..., 1], 0, tile_height)
    tile_maxs[..., 0] = torch.clamp(tile_maxs[..., 0], 0, tile_width)
    tile_maxs[..., 1] = torch.clamp(tile_maxs[..., 1], 0, tile_height)
    tiles_per_gauss = (tile_maxs - tile_mins).prod(dim=-1)  # [C, N]
    tiles_per_gauss *= radii > 0.0

    n_isects = tiles_per_gauss.sum().item()
    isect_ids = torch.empty(n_isects, dtype=torch.int64, device=device)
    flatten_ids = torch.empty(n_isects, dtype=torch.int32, device=device)

    cum_tiles_per_gauss = torch.cumsum(tiles_per_gauss.flatten(), dim=0)
    tile_n_bits = (tile_width * tile_height).bit_length()

    def binary(num):
        return "".join("{:0>8b}".format(c) for c in struct.pack("!f", num))

    def kernel(cam_id, gauss_id):
        if radii[cam_id, gauss_id] <= 0.0:
            return
        index = cam_id * N + gauss_id
        curr_idx = cum_tiles_per_gauss[index - 1] if index > 0 else 0

        depth_id = struct.unpack("i", struct.pack("f", depths[cam_id, gauss_id]))[0]

        tile_min = tile_mins[cam_id, gauss_id]
        tile_max = tile_maxs[cam_id, gauss_id]
        for y in range(tile_min[1], tile_max[1]):
            for x in range(tile_min[0], tile_max[0]):
                tile_id = y * tile_width + x
                isect_ids[curr_idx] = (
                    (cam_id << 32 << tile_n_bits) | (tile_id << 32) | depth_id
                )
                flatten_ids[curr_idx] = index  # flattened index
                curr_idx += 1

    for cam_id in range(C):
        for gauss_id in range(N):
            kernel(cam_id, gauss_id)

    if sort:
        isect_ids, sort_indices = torch.sort(isect_ids)
        flatten_ids = flatten_ids[sort_indices]

    return tiles_per_gauss.int(), isect_ids, flatten_ids


@torch.no_grad()
def _isect_offset_encode(
    isect_ids: Tensor, C: int, tile_width: int, tile_height: int
) -> Tensor:
    """Pytorch implementation of `gsplat.cuda._wrapper.isect_offset_encode()`.

    .. note::

        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    tile_n_bits = (tile_width * tile_height).bit_length()
    tile_counts = torch.zeros(
        (C, tile_height, tile_width), dtype=torch.int64, device=isect_ids.device
    )

    isect_ids_uq, counts = torch.unique_consecutive(isect_ids >> 32, return_counts=True)

    cam_ids_uq = isect_ids_uq >> tile_n_bits
    tile_ids_uq = isect_ids_uq & ((1 << tile_n_bits) - 1)
    tile_ids_x_uq = tile_ids_uq % tile_width
    tile_ids_y_uq = tile_ids_uq // tile_width

    tile_counts[cam_ids_uq, tile_ids_y_uq, tile_ids_x_uq] = counts

    cum_tile_counts = torch.cumsum(tile_counts.flatten(), dim=0).reshape_as(tile_counts)
    offsets = cum_tile_counts - tile_counts
    return offsets.int()


def accumulate(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    opacities: Tensor,  # [C, N]
    colors: Tensor,  # [C, N, channels]
    gaussian_ids: Tensor,  # [M]
    pixel_ids: Tensor,  # [M]
    camera_ids: Tensor,  # [M]
    image_width: int,
    image_height: int,
) -> Tuple[Tensor, Tensor]:
    """Alpah compositing of 2D Gaussians in Pure Pytorch.

    This function performs alpha compositing for Gaussians based on the pair of indices
    {gaussian_ids, pixel_ids, camera_ids}, which annotates the intersection between all
    pixels and Gaussians. These intersections can be accquired from
    `gsplat.rasterize_to_indices_in_range`.

    .. note::

        This function exposes the alpha compositing process into pure Pytorch.
        So it relies on Pytorch's autograd for the backpropagation. It is much slower
        than our fully fused rasterization implementation and comsumes much more GPU memory.
        But it could serve as a playground for new ideas or debugging, as no backward
        implementation is needed.

    .. warning::

        This function requires the `nerfacc` package to be installed. Please install it
        using the following command `pip install nerfacc`.

    Args:
        means2d: Gaussian means in 2D. [C, N, 2]
        conics: Inverse of the 2D Gaussian covariance, Only upper triangle values. [C, N, 3]
        opacities: Per-view Gaussian opacities (for example, when antialiasing is
            enabled, Gaussian in each view would efficiently have different opacity). [C, N]
        colors: Per-view Gaussian colors. Supports N-D features. [C, N, channels]
        gaussian_ids: Collection of Gaussian indices to be rasterized. A flattened list of shape [M].
        pixel_ids: Collection of pixel indices (row-major) to be rasterized. A flattened list of shape [M].
        camera_ids: Collection of camera indices to be rasterized. A flattened list of shape [M].
        image_width: Image width.
        image_height: Image height.

    Returns:
        A tuple:

        - **renders**: Accumulated colors. [C, image_height, image_width, channels]
        - **alphas**: Accumulated opacities. [C, image_height, image_width, 1]
    """

    try:
        from nerfacc import accumulate_along_rays, render_weight_from_alpha
    except ImportError:
        raise ImportError("Please install nerfacc package: pip install nerfacc")

    C, N = means2d.shape[:2]
    channels = colors.shape[-1]

    pixel_ids_x = pixel_ids % image_width
    pixel_ids_y = pixel_ids // image_width
    pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1) + 0.5  # [M, 2]
    deltas = pixel_coords - means2d[camera_ids, gaussian_ids]  # [M, 2]
    c = conics[camera_ids, gaussian_ids]  # [M, 3]
    sigmas = (
        0.5 * (c[:, 0] * deltas[:, 0] ** 2 + c[:, 2] * deltas[:, 1] ** 2)
        + c[:, 1] * deltas[:, 0] * deltas[:, 1]
    )  # [M]
    alphas = torch.clamp_max(
        opacities[camera_ids, gaussian_ids] * torch.exp(-sigmas), 0.999
    )

    indices = camera_ids * image_height * image_width + pixel_ids
    total_pixels = C * image_height * image_width

    weights, trans = render_weight_from_alpha(
        alphas, ray_indices=indices, n_rays=total_pixels
    )
    renders = accumulate_along_rays(
        weights,
        colors[camera_ids, gaussian_ids],
        ray_indices=indices,
        n_rays=total_pixels,
    ).reshape(C, image_height, image_width, channels)
    alphas = accumulate_along_rays(
        weights, None, ray_indices=indices, n_rays=total_pixels
    ).reshape(C, image_height, image_width, 1)

    return renders, alphas


def _rasterize_to_pixels(
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    colors: Tensor,  # [C, N, channels]
    opacities: Tensor,  # [C, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    batch_per_iter: int = 100,
):
    """Pytorch implementation of `gsplat.cuda._wrapper.rasterize_to_pixels()`.

    This function rasterizes 2D Gaussians to pixels in a Pytorch-friendly way. It
    iteratively accumulates the renderings within each batch of Gaussians. The
    interations are controlled by `batch_per_iter`.

    .. note::
        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.

    .. note::

        This function relies on Pytorch's autograd for the backpropagation. It is much slower
        than our fully fused rasterization implementation and comsumes much more GPU memory.
        But it could serve as a playground for new ideas or debugging, as no backward
        implementation is needed.

    .. warning::

        This function requires the `nerfacc` package to be installed. Please install it
        using the following command `pip install nerfacc`.
    """
    from ._wrapper import rasterize_to_indices_in_range

    C, N = means2d.shape[:2]
    n_isects = len(flatten_ids)
    device = means2d.device

    render_colors = torch.zeros(
        (C, image_height, image_width, colors.shape[-1]), device=device
    )
    render_alphas = torch.zeros((C, image_height, image_width, 1), device=device)

    # Split Gaussians into batches and iteratively accumulate the renderings
    block_size = tile_size * tile_size
    isect_offsets_fl = torch.cat(
        [isect_offsets.flatten(), torch.tensor([n_isects], device=device)]
    )
    max_range = (isect_offsets_fl[1:] - isect_offsets_fl[:-1]).max().item()
    num_batches = (max_range + block_size - 1) // block_size
    for step in range(0, num_batches, batch_per_iter):
        transmittances = 1.0 - render_alphas[..., 0]

        # Find the M intersections between pixels and gaussians.
        # Each intersection corresponds to a tuple (gs_id, pixel_id, camera_id)
        gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range(
            step,
            step + batch_per_iter,
            transmittances,
            means2d,
            conics,
            opacities,
            image_width,
            image_height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )  # [M], [M]
        if len(gs_ids) == 0:
            break

        # Accumulate the renderings within this batch of Gaussians.
        renders_step, accs_step = accumulate(
            means2d,
            conics,
            opacities,
            colors,
            gs_ids,
            pixel_ids,
            camera_ids,
            image_width,
            image_height,
        )
        render_colors = render_colors + renders_step * transmittances[..., None]
        render_alphas = render_alphas + accs_step * transmittances[..., None]

    render_alphas = render_alphas
    if backgrounds is not None:
        render_colors = render_colors + backgrounds[:, None, None, :] * (
            1.0 - render_alphas
        )

    return render_colors, render_alphas


def _eval_sh_bases_fast(basis_dim: int, dirs: Tensor):
    """
    Evaluate spherical harmonics bases at unit direction for high orders
    using approach described by
    Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
    https://jcgt.org/published/0002/02/06/


    :param basis_dim: int SH basis dim. Currently, only 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)

    See reference C++ code in https://jcgt.org/published/0002/02/06/code.zip
    """
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )

    result[..., 0] = 0.2820947917738781

    if basis_dim <= 1:
        return result

    x, y, z = dirs.unbind(-1)

    fTmpA = -0.48860251190292
    result[..., 2] = -fTmpA * z
    result[..., 3] = fTmpA * x
    result[..., 1] = fTmpA * y

    if basis_dim <= 4:
        return result

    z2 = z * z
    fTmpB = -1.092548430592079 * z
    fTmpA = 0.5462742152960395
    fC1 = x * x - y * y
    fS1 = 2 * x * y
    result[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    result[..., 7] = fTmpB * x
    result[..., 5] = fTmpB * y
    result[..., 8] = fTmpA * fC1
    result[..., 4] = fTmpA * fS1

    if basis_dim <= 9:
        return result

    fTmpC = -2.285228997322329 * z2 + 0.4570457994644658
    fTmpB = 1.445305721320277 * z
    fTmpA = -0.5900435899266435
    fC2 = x * fC1 - y * fS1
    fS2 = x * fS1 + y * fC1
    result[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    result[..., 13] = fTmpC * x
    result[..., 11] = fTmpC * y
    result[..., 14] = fTmpB * fC1
    result[..., 10] = fTmpB * fS1
    result[..., 15] = fTmpA * fC2
    result[..., 9] = fTmpA * fS2

    if basis_dim <= 16:
        return result

    fTmpD = z * (-4.683325804901025 * z2 + 2.007139630671868)
    fTmpC = 3.31161143515146 * z2 - 0.47308734787878
    fTmpB = -1.770130769779931 * z
    fTmpA = 0.6258357354491763
    fC3 = x * fC2 - y * fS2
    fS3 = x * fS2 + y * fC2
    result[..., 20] = 1.984313483298443 * z2 * (
        1.865881662950577 * z2 - 1.119528997770346
    ) + -1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201)
    result[..., 21] = fTmpD * x
    result[..., 19] = fTmpD * y
    result[..., 22] = fTmpC * fC1
    result[..., 18] = fTmpC * fS1
    result[..., 23] = fTmpB * fC2
    result[..., 17] = fTmpB * fS2
    result[..., 24] = fTmpA * fC3
    result[..., 16] = fTmpA * fS3
    return result


def _spherical_harmonics(
    degree: int,
    dirs: torch.Tensor,  # [..., 3]
    coeffs: torch.Tensor,  # [..., K, 3]
):
    """Pytorch implementation of `gsplat.cuda._wrapper.spherical_harmonics()`."""
    dirs = F.normalize(dirs, p=2, dim=-1)
    num_bases = (degree + 1) ** 2
    bases = torch.zeros_like(coeffs[..., 0])
    bases[..., :num_bases] = _eval_sh_bases_fast(num_bases, dirs)
    return (bases[..., None] * coeffs).sum(dim=-2)

def generate_rays(
    c2w: Tensor, 
    K: Tensor, 
    width: int, 
    height: int,
) -> Tuple[Tensor, Tensor]:

    i, j = torch.meshgrid(torch.arange(width, device=K.device),
                          torch.arange(height, device=K.device),
                          indexing="ij")
    i = i.t().float()
    j = j.t().float()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    directions = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], dim=-1)  # [H, W, 3]
    directions = directions @ c2w[:3, :3].T  
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)  

    origins = c2w[:3, 3].expand_as(directions)

    return origins, directions

from scipy.stats import chi2
import math
from scipy.spatial.transform import Rotation as R
def rotmat_to_quat(R_tensor: torch.Tensor) -> torch.Tensor:
    R_np = R_tensor.cpu().numpy()  # shape: (N, 3, 3)
    
    rot_obj = R.from_matrix(R_np)
    quats_np = rot_obj.as_quat()  
    
    quats_np = np.concatenate([quats_np[:, -1:], quats_np[:, :-1]], axis=1)
    
    quats_tensor = torch.from_numpy(quats_np).to(R_tensor.device, dtype=R_tensor.dtype)  # shape: (N, 4)
    return quats_tensor

def normals_to_quats(normals: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:

    N = normals.shape[0]
    candidate = torch.zeros_like(normals)
    candidate[:, 0] = 1.0
    mask = (torch.abs(normals[:, 0]) > 0.9)
    candidate[mask] = torch.tensor([0.0, 1.0, 0.0], device=normals.device)
    
    tangent1 = torch.cross(normals, candidate, dim=1)
    tangent1 = tangent1 / (torch.norm(tangent1, dim=1, keepdim=True) + eps)
    
    tangent2 = torch.cross(normals, tangent1, dim=1)
    tangent2 = tangent2 / (torch.norm(tangent2, dim=1, keepdim=True) + eps)
    
    R = torch.stack([tangent1, tangent2, normals], dim=2)  # shape: [N, 3, 3]
    
    quats = rotmat_to_quat(R)  # shape: [N, 4]
    return quats

def filter_scales(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sdf_coeffs: torch.Tensor,
    alpha_thresh: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    final_mask = (scales > alpha_thresh).all(dim=1)

    return means[final_mask], quats[final_mask], scales[final_mask], opacities[final_mask], sdf_coeffs[final_mask]


def gaussian_to_ellipse(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    # covs, _ = _quat_scale_to_covar_preci(quats, scales) 
    # U, S, Vh = torch.linalg.svd(covs)

    # S_reduced = S[:, :2]  # Select the two largest eigenvalues [N, 2]
    # U_reduced = U[:, :, :2] 

    # r_a = torch.sqrt(S_reduced[:, 0])  
    # r_b = torch.sqrt(S_reduced[:, 1])

    # axes_a = U_reduced[:, :, 0] / torch.norm(U_reduced[:, :, 0], dim=-1, keepdim=True)  
    # axes_b = U_reduced[:, :, 1] / torch.norm(U_reduced[:, :, 1], dim=-1, keepdim=True) 
    
    # return r_a, r_b, axes_a, axes_b

    U = _quat_to_rotmat(quats) # shape: (N, 3, 3)
    S = scales**2               # shape: (N, 3)

    sorted_S, indices = torch.sort(S, dim=1, descending=True)
    U_sorted = torch.gather(U, dim=2, index=indices.unsqueeze(1).expand(-1, 3, -1))

    # opacities_np = opacities.detach().cpu().numpy()
    # scale_factor_np = np.sqrt(chi2.ppf(0.99, df=3)) * opacities_np
    # scale_factor = torch.tensor(scale_factor_np, device=opacities.device)
    r_a = torch.sqrt(sorted_S[:, 0])# * scale_factor
    r_b = torch.sqrt(sorted_S[:, 1])# * scale_factor 

    axes_a = U_sorted[:, :, 0] / torch.norm(U_sorted[:, :, 0], dim=-1, keepdim=True)  
    axes_b = U_sorted[:, :, 1] / torch.norm(U_sorted[:, :, 1], dim=-1, keepdim=True)
   
    n_vec = torch.cross(axes_a, axes_b, dim=-1)  # [N, 3]
    n_vec = n_vec / torch.norm(n_vec, dim=-1, keepdim=True)  # Normalize the normal vector


    return r_a, r_b, axes_a, axes_b, n_vec


def cull_mask(
    ro: Tensor, # [3]
    rd: Tensor, # [3]
    r_a: Tensor,
    r_b: Tensor, 
    means: Tensor, # [N,3]
) -> Tensor:

    v = means - ro
    
    in_front = torch.sum(v * rd, dim=-1) > 0

    dist = torch.norm(torch.cross(v, rd.expand_as(v), dim=-1), dim=-1)  # [N, 3]

    # return (dist <= r_a) | (dist <= r_b)
    return in_front & ((dist <= r_a) | (dist <= r_b))

import numpy as np
from scipy.spatial import cKDTree


def sdf_harmonics(
    degree: int, 
    dirs: torch.Tensor, 
    coeffs: torch.Tensor
):
    dirs = F.normalize(dirs, p=2, dim=-1)
    num_bases = (degree + 1) ** 2
    bases = torch.zeros_like(coeffs)
    bases[..., :num_bases] = _eval_sh_bases_fast(num_bases, dirs)
    return (bases * coeffs).sum(dim=-1)

def knn_candidates(
    means, 
    points, 
    k=20
):
    means_np = means.detach().cpu().numpy()
    points_np = points.detach().cpu().numpy()

    tree = cKDTree(means_np)
    dist, indices = tree.query(points_np, k=k)

    return torch.tensor(dist, device=means.device, dtype=torch.float32), torch.tensor(indices, device=means.device, dtype=torch.long)

def safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    return torch.sqrt((x ** 2).sum(dim=dim).clamp_min(eps))

def signed_distance_knn(
    pos: torch.Tensor,    # [M, 3]
    means: torch.Tensor,  # [N, 3]
    r_a: torch.Tensor,    # [N]
    r_b: torch.Tensor,    # [N]
    axes_a: torch.Tensor, # [N, 3]
    axes_b: torch.Tensor, # [N, 3],
    sdf_coeffs: torch.Tensor, # [N, K]
    sh_degree: int,
    k: int = 10,
    eps = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    _, indices = knn_candidates(means, pos, k=k)  # [M, k]
    means_knn   = means[indices]      # [M, k, 3]
    r_a_knn     = r_a[indices]        # [M, k]
    r_b_knn     = r_b[indices]        # [M, k]
    axes_a_knn  = axes_a[indices]     # [M, k, 3]
    axes_b_knn  = axes_b[indices]     # [M, k, 3]

    diff = pos.unsqueeze(1) - means_knn  # [M, k, 3]

    proj_a = torch.sum(diff * axes_a_knn, dim=-1)  # [M, k]
    proj_b = torch.sum(diff * axes_b_knn, dim=-1)  # [M, k]
    proj_point = proj_a.unsqueeze(-1) * axes_a_knn + proj_b.unsqueeze(-1) * axes_b_knn  # [M, k, 3]

    scaled_a = proj_a / r_a_knn.clamp_min(eps)  # [M, k]
    scaled_b = proj_b / r_b_knn.clamp_min(eps)  # [M, k]
    scale_factor = torch.sqrt(scaled_a**2 + scaled_b**2 + eps)  # [M, k]

    closest_a = proj_a / scale_factor.clamp_min(eps)  # [M, k]
    closest_b = proj_b / scale_factor.clamp_min(eps)  # [M, k]
    closest_point_boundary = (closest_a.unsqueeze(-1) * axes_a_knn +
                              closest_b.unsqueeze(-1) * axes_b_knn)  # [M, k, 3]
    
    dist_proj = safe_norm(diff - proj_point, dim=-1, eps=eps)
    dist_boundary = safe_norm(diff - closest_point_boundary, dim=-1, eps=eps) 

    inside_mask = (scaled_a**2 + scaled_b**2) <= 1  # [M, k]
    dist = torch.where(inside_mask, dist_proj, dist_boundary)  # [M, k]

    _, idx = torch.min(dist, dim=1)  # idx: [M]
    orig_idx = indices[torch.arange(indices.shape[0]), idx]  # [M]

    selected_coeffs = sdf_coeffs[orig_idx]  # [M, K]

    selected_means = means_knn[torch.arange(means_knn.shape[0]), idx]  # [M, 3]
    
    dirs = selected_means - pos  # [M, 3]
    depth_harmonics = sdf_harmonics(sh_degree, dirs, selected_coeffs)  # [M]
    
    sdf_base = dist[torch.arange(dist.shape[0]), idx]  # [M]
    dist_min = sdf_base + depth_harmonics

    return dist_min, orig_idx

def update_id_to_count(id_to_count, gaussian_ids, orig_idx):
    gaussian_ids_list = sorted(list(gaussian_ids))

    # Then convert the list to a tensor.
    gaussian_ids_tensor = torch.tensor(gaussian_ids_list, dtype=torch.long, device=orig_idx.device)


    orig_idx = orig_idx.view(-1).long()  
    if orig_idx.numel() == 0:
        return id_to_count 
    
    if orig_idx.max() >= gaussian_ids_tensor.size(0):
        print(f"orig_idx: {orig_idx.max()}, gaussian_ids: {gaussian_ids_tensor.size(0)}")
        raise ValueError("orig_idx contains an index out of bounds for gaussian_ids")
    
    closest_ids = gaussian_ids_tensor[orig_idx]  
    unique_ids, counts = torch.unique(closest_ids, return_counts=True)
    
    for uid, cnt in zip(unique_ids.tolist(), counts.tolist()):
        id_to_count[uid] = id_to_count.get(uid, 0) + cnt
    
    return id_to_count



from typing import Callable

def hessian_loss(
    pos: Tensor,
    means: Tensor,
    normals: Tensor,
    sdf_fn: Callable[[Tensor], Tensor],
    # sdf_weight: float = 1e-2,
    # hessian_weight: float = 1e-8,
    eps: float = 1e-8
) -> Tuple[Tensor, Tensor]:
    """
    Combined Hessian and L1 SDF loss.
    """
    pos = pos.clone().detach().requires_grad_(True)

    # Evaluate SDF values
    sdf, idx = sdf_fn(pos)  # [M]

    #First-order gradients
    # grad_sdf = torch.autograd.grad(
    #     outputs=sdf,
    #     inputs=pos,
    #     grad_outputs=torch.ones_like(sdf),
    #     create_graph=True,
    #     retain_graph=True,
    #     only_inputs=True
    # )[0]  # [M, 3]

    # # Second-order (Hessian)
    # hessians = []
    # for i in range(grad_sdf.shape[-1]):
    #     grad2 = torch.autograd.grad(
    #         outputs=grad_sdf[:, i],
    #         inputs=pos,
    #         grad_outputs=torch.ones_like(grad_sdf[:, i]),
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True
    #     )[0]
    #     hessians.append(grad2)

    # hessian = torch.stack(hessians, dim=2)  # [M, 3, 3]
    # hessian_squared_norm = (hessian ** 2).sum(dim=(1, 2)).clamp_min(eps)  # [M]

    # hessian_loss = hessian_squared_norm.mean()
    sdf_loss = torch.abs(sdf).mean()  # same as F.l1_loss(sdf, 0)

    # Gather the normal for each point based on idx
    n_i = normals[idx]       # shape: [M, 3]
    mu_i = means[idx]        # shape: [M, 3]
    
    # eps = torch.sum((pos - mu_i) * mu_i, dim=-1) 
    
    sdf_sum = sdf.sum()  
    grad = torch.autograd.grad(
        outputs=sdf_sum,
        inputs=pos,
        create_graph=True,
        retain_graph=True
    )[0]  # grad: [M, 3]
    
    grad_norm = grad.norm(dim=-1, keepdim=True) + 1e-8  # [M, 1]
    cos_term = torch.sum(grad * n_i, dim=-1, keepdim=True) / grad_norm  # [M, 1]

    # L^v_cons: encourage the SDF gradient to be aligned with the surfel normal.
    v_cons = torch.mean(1.0 - cos_term)


    # # ---- Distance Consistency Term ----
    # # offset e_i = dot( (p_i - mu_i), n_i )
    # #   p_i = pos[i],  mu_i = means[idx[i]],  n_i = normals[idx[i]]
    # offset_i = (pos - mu_i) * n_i
    # offset_i = offset_i.sum(dim=-1)  # shape: [M]

    # # L^d_cons = mean( |SDF - offset_i| )
    # dist_cons = (sdf - offset_i).abs().mean()

    # # ---- Normal Consistency Term ----
    # # We want 1 - cos(grad_sdf, n_i) = 1 - (grad_sdf·n_i / ||grad_sdf||)
    # # Make sure the SDF gradient is normalized before the dot product
    # grad_norm = grad_sdf.norm(dim=-1, keepdim=True).clamp_min(eps)
    # grad_sdf_unit = grad_sdf / grad_norm

    # # Dot product with normal n_i (assuming n_i is already normalized)
    # dot_vals = (grad_sdf_unit * n_i).sum(dim=-1)  # shape: [M]
    # normal_cons = (1.0 - dot_vals).mean()

    return sdf_loss, v_cons, idx

def compute_hessian_frobenius(f_vals, x):
    # f_vals: [M] , x: [M,3], both require_grad=True
    grads = torch.autograd.grad(f_vals.sum(), x, create_graph=True)[0]  # [M,3]
    H = []
    for i in range(3):
        # ∂²f/∂xᵢ∂x
        dgi = torch.autograd.grad(grads[:, i].sum(), x, create_graph=True)[0]  # [M,3]
        H.append(dgi)
    # stack along new dim → [3, M, 3], then permute → [M,3,3]
    H = torch.stack(H, dim=0).permute(1,0,2)
    # Frobenius norm² of each Hessian
    return (H.pow(2).sum(dim=(-2,-1))).mean()  # scalar

def surface_consistency_loss(
    pos: torch.Tensor,         # [M, 3]: 
    means: torch.Tensor,         # [N, 3]:
    global_normals: torch.Tensor,     # [N, 3]: 
    sdf_fn: Callable[[torch.Tensor], torch.Tensor],  
    #gaussian_ids: torch.Tensor,  # [M]:
    eps_max: float = 0.05,
    retain_graph: bool = True 
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    device = pos.device

    
    pos = pos.clone().detach().requires_grad_(True)

    
    sdf, idx = sdf_fn(pos)  # sdf_vals: [M]
    L_sdf = (torch.abs(sdf)).mean()

    # closest_ids = gaussian_ids[idx]  # [M]
    # gidx, counts = torch.unique(closest_ids, return_counts=True)

    # selected_means = means[gidx]  # shape: [M, 3]
    # selected_normals = global_normals[gidx]  # shape: [M, 3]


    M = global_normals.shape[0]

    epsilons = torch.empty(M, device=device).uniform_(-eps_max, eps_max)  # [M]
    
    query_points = pos + epsilons.unsqueeze(-1) * global_normals  # [M, 3]
    query_points = query_points.clone().detach().requires_grad_(True)  # [M, 3]

    sdf_vals, _ = sdf_fn(query_points)  # [M]
    
    sdf_sum = sdf_vals.sum()  # scalar to accumulate contributions
    grads = torch.autograd.grad(
        outputs=sdf_sum,
        inputs=query_points,
        create_graph=True,
        retain_graph=retain_graph
    )[0]  # grads: [M, 3]

    L_d = (torch.abs(sdf_vals - epsilons)).mean()  # L^d_cons

    grad_norm = torch.norm(grads, dim=-1, keepdim=True) + 1e-12
    cos_sim = torch.sum(grads * global_normals, dim=-1, keepdim=True) / grad_norm
    L_v = torch.mean(1.0 - cos_sim)    

    L_hess = torch.zeros_like(L_sdf, device=device)

    return L_sdf, L_d, L_v, L_hess, sdf, idx

def unproject_depths(depth, mask, w2c, K):
    # grab device/dtype from depth
    device = depth.device
    dtype  = depth.dtype

    depth = depth.unsqueeze(0).unsqueeze(0)
    # 2) interpolate by factor 0.25 in each spatial dim
    depth = F.interpolate(
        depth,
        scale_factor=(0.25, 0.25),
        mode='nearest',
        recompute_scale_factor=True
    )
    # 3) remove the extra dims → (H', W')
    depth = depth.squeeze(0).squeeze(0)
    print(depth.shape)

    mask = mask.to(device)
    w2c  = w2c.to(device)
    K    = K.to(device)

    depth_h, depth_w = depth.shape  # (128, 160)
    image_w, image_h = mask.shape   # (640, 512)

    scale_x = image_w / depth_w  # 4
    scale_y = image_h / depth_h  # 4

    v_depth, u_depth = torch.meshgrid(
        torch.arange(depth_h, device=device, dtype=dtype),
        torch.arange(depth_w, device=device, dtype=dtype),
        indexing='ij'
    )

    u_img = u_depth * scale_x
    v_img = v_depth * scale_y

    u_flat  = u_img.flatten()
    v_flat  = v_img.flatten()
    z_flat  = depth.flatten()

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    X = (u_flat - cx) * z_flat / fx
    Y = (v_flat - cy) * z_flat / fy
    Z = z_flat
    pts_cam = torch.stack([X, Y, Z], dim=-1)   # (N,3)

    ones = torch.ones((pts_cam.shape[0], 1), device=device, dtype=dtype)
    pts_cam_h = torch.cat([pts_cam, ones], dim=-1)   # (N,4)

    w2c_inv = torch.linalg.inv(w2c)
    pts_w_h = (w2c_inv @ pts_cam_h.T).T            # (N,4)
    pts_w   = pts_w_h[:, :3] / pts_w_h[:, 3:4]     # (N,3)

    u_idx = torch.round(u_flat).long().clamp(0, image_w - 1)
    v_idx = torch.round(v_flat).long().clamp(0, image_h - 1)

    # 12) boolean mask pick
    keep = mask[v_idx, u_idx]                     # still torch.bool
    return pts_w[keep]

def eikonal_loss(
    pos: torch.Tensor,
    sdf: torch.Tensor,
) -> torch.Tensor:
    # Compute the gradient of the SDF with respect to points.
    grad_sdf = torch.autograd.grad(
        outputs=sdf,
        inputs=pos,
        grad_outputs=torch.ones_like(sdf),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # shape: [M, 3]
    
    # Enforce that the gradient norm is close to 1.
    loss = ((grad_sdf.norm(dim=-1) - 1) ** 2).mean()
    return loss

def laplacian_loss(pos, sdf):
    # Compute first derivatives
    grad_sdf = torch.autograd.grad(
        outputs=sdf,
        inputs=pos,
        grad_outputs=torch.ones_like(sdf),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # [M, 3]
    
    laplacian = 0.0
    # Sum the second derivatives along the diagonal to compute the Laplacian.
    for i in range(pos.shape[-1]):
        grad2 = torch.autograd.grad(
            outputs=grad_sdf[:, i],
            inputs=pos,
            grad_outputs=torch.ones_like(grad_sdf[:, i]),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0][:, i]  # [M]
        laplacian += grad2
    
    loss = (laplacian ** 2).mean()
    return loss


def signed_distance(
    pos: torch.Tensor,    # [M, 3]
    means: torch.Tensor,  # [N, 3]
    r_a: torch.Tensor,    # [N]
    r_b: torch.Tensor,    # [N]
    axes_a: torch.Tensor, # [N, 3]
    axes_b: torch.Tensor, # [N, 3]
) -> Tuple[torch.Tensor, torch.Tensor]:

    diff = pos[:, None, :] - means[None, :, :]

    proj_a = torch.sum(diff * axes_a[None, :, :], dim=-1)  # [M,N]
    proj_b = torch.sum(diff * axes_b[None, :, :], dim=-1)  # [M,N]
    
    proj_point = (
        proj_a.unsqueeze(-1) * axes_a[None, :, :] +
        proj_b.unsqueeze(-1) * axes_b[None, :, :]
    )  # [M,N,3]
    
    scaled_a = proj_a / (r_a[None, :])  # [M,N]
    scaled_b = proj_b / (r_b[None, :])  # [M,N]
    
    scale_factor = torch.sqrt(scaled_a**2 + scaled_b**2 + 1e-8)  # [M,N]
    
    closest_a = proj_a / (scale_factor)  # [M,N]
    closest_b = proj_b / (scale_factor) # [M,N]
    closest_point_boundary = (
        closest_a.unsqueeze(-1) * axes_a[None, :, :] +
        closest_b.unsqueeze(-1) * axes_b[None, :, :]
    )  # [M,N,3]
    
    dist_proj = torch.norm(diff - proj_point, dim=-1)  # [M,N]
    dist_boundary = torch.norm(diff - closest_point_boundary, dim=-1)  # [M,N]

    inside_mask = (scaled_a**2 + scaled_b**2) <= 1  # [M,N]

    dist = torch.where(inside_mask, dist_proj, dist_boundary)  # [M,N]
    
    dist_min, dist_indices = torch.min(dist, dim=1)  # dist_min: [M], dist_indices: [M]

    return dist_min, dist_indices



def sphere_trace(
    means: Tensor,
    r_a: Tensor,
    r_b: Tensor,
    axes_a: Tensor,
    axes_b: Tensor,
    sdf_coeffs: Tensor,
    sh_degree: int,
    origins: Tensor,
    directions: Tensor,
    max_steps: int = 526,
    min_hit_distance: float = 0.0001,
    max_trace_distance: float = 10.0,
):
    H, W = origins.shape[:2]
    num_rays = H*W

    ro = origins.view(num_rays, 3)
    rd = directions.view(num_rays, 3)

    alive     = torch.ones(num_rays, dtype=torch.bool, device=ro.device)
    t         = torch.zeros(num_rays,      device=ro.device)
    hit_dist  = torch.full((num_rays,), max_trace_distance, device=ro.device)

    for _ in range(max_steps):
        if not alive.any():
            break

        idx_alive = alive.nonzero(as_tuple=False).squeeze(1)
        pos       = ro[idx_alive] + t[idx_alive,None] * rd[idx_alive]
        dist_alive, _ = signed_distance_knn(
            pos, means, r_a, r_b, axes_a, axes_b, sdf_coeffs, sh_degree
        )

        hit_here = dist_alive < min_hit_distance

        if hit_here.any():
            hit_indices_global = idx_alive[hit_here]
            hit_dist[hit_indices_global] = t[hit_indices_global]
            alive[hit_indices_global] = False

        t[idx_alive] = t[idx_alive] + dist_alive

        too_far = t > max_trace_distance
        if too_far.any():
            hit_dist[too_far] = max_trace_distance
            alive[too_far]    = False

    return hit_dist.view(H, W)


    

def sphere_trace_old(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    colors: torch.Tensor,
    origins: Tensor,
    directions: Tensor,
    max_steps: int = 256,
    min_hit_distance: float = 0.001,
    max_trace_distance: float = 10.0,
) -> torch.Tensor:

    H, W = origins.shape[:2]
    N = means.shape[0]

    total_distance = torch.zeros((H, W), device=means.device)
    hit_color = torch.zeros((H, W, 3), device=means.device)

    r_a, r_b, axes_a, axes_b = gaussian_to_ellipse(means, quats, scales)


    for i in range(H):
        for j in range(W):

            ro = origins[i,j]
            rd = directions[i, j]
            
            mask = cull_mask(ro, rd, r_a, r_b, means)

            r_a_reduced = r_a[mask]
            r_b_reduced = r_b[mask]
            axes_a_reduced = axes_a[mask]
            axes_b_reduced = axes_b[mask]
            means_reduced = means[mask]
            colors_reduced = colors[mask]
            print(f"pixel ({i, j}) / ({H, W}), number of gaussians {means_reduced.shape[0]}", end="\r")

            if means_reduced.shape[0] == 0:
                total_distance[i, j] = max_trace_distance
                continue

            t = 0.0
            ray_hit_color = torch.zeros(3, device=means.device)
            for steps in range(max_steps):
                pos = ro + t*rd

                dist, idx = signed_distance(pos, means_reduced, r_a_reduced, r_b_reduced, axes_a_reduced, axes_b_reduced)
                
                if dist < min_hit_distance:
                    ray_hit_color = colors_reduced[idx, 0]
                    total_distance[i, j] = t
                    break
                
                t += dist

                if dist > max_trace_distance:
                    total_distance[i, j] = max_trace_distance
                    break
                
            # total_distance[i, j] = t
            hit_color[i, j] = ray_hit_color

    return hit_color, total_distance

def calculate_depth(
    K: Tensor,
    distance: Tensor,
    width: int, 
    height: int,
) -> Tensor:

    K_inv = torch.linalg.inv(K)

    i, j = torch.meshgrid(
        torch.arange(width, device=K.device),
        torch.arange(height, device=K.device),
        indexing="ij"
    )

    pixels_homog = torch.stack([i, j, torch.ones_like(i)], dim=-1).float()
    pixels_normalized = (K_inv @ pixels_homog.reshape(-1, 3).T).T.reshape(height, width, 3)  # [H, W, 3]

    denom = torch.sqrt(pixels_normalized[..., 0]**2 + pixels_normalized[..., 1]**2 + 1)
    depth = distance / denom

    return depth

def generate_depth_image(
    depth: Tensor,
    distance: Tensor,
    distance_threshold: float,
) -> Tensor:

    threshold_mask = distance < distance_threshold 

    min_depth = 0.0
    max_depth = depth[threshold_mask].max() if threshold_mask.any() else 1.0
    depth_norm = torch.zeros_like(depth)

    depth_norm[threshold_mask] = (depth[threshold_mask] - min_depth) / (max_depth - min_depth)

    return depth_norm.unsqueeze(-1).expand(-1, -1, 3) 


from sklearn.neighbors import KDTree
import numpy as np

@torch.no_grad()
def outer_ellipsoid(
    points: Tensor,
    rgbs: Tensor = None,
    sample_size: int = None,
    tol: float = 0.001,
    max_iter: int = 5000,
    k: int = 5,
    max_dist: float = 0.01,
    device = "cuda",
) -> Tuple[Tensor, Tensor, Tensor]:
    
    points_np = points.cpu().numpy()
    tree = KDTree(points_np)
    if sample_size is not None:
        sample = np.random.choice(points_np.shape[0], size=sample_size, replace=False)
        sampled_points = points_np[sample]
        dist, idx = tree.query(sampled_points, k=k)
    else:
        dist, idx = tree.query(points_np, k=k)

    valid_neighbors = np.all(dist <= max_dist, axis=1)
    valid_idx = idx[valid_neighbors]

    clusters = torch.tensor(points_np[valid_idx], dtype=torch.float64, device=device)
    
    B, N, d = clusters.shape
    Q = torch.cat((clusters, torch.ones((B, N, 1), dtype=torch.float64, device=device)), dim=2).permute(0, 2, 1)
    u = torch.ones((B, N), dtype=torch.float64, device=device) / N  # [B, N]
    err = torch.ones(B, dtype=torch.float64, device=device) * (1 + tol)
    active_mask = err > tol  

    for i in range(max_iter):
        if not torch.any(active_mask):
            break

        X = Q[active_mask] @ torch.diag_embed(u[active_mask]) @ Q[active_mask].transpose(1, 2)  # [B_active, 4, 4]
        X_inv = torch.linalg.inv(X)  # [B_active, 4, 4]
        M = torch.diagonal(Q[active_mask].transpose(1, 2) @ X_inv @ Q[active_mask], dim1=1, dim2=2)  # [B_active, N]
        jdx = torch.argmax(M, dim=1)  # [B_active,]
        step_size = (M[torch.arange(M.shape[0]), jdx] - d - 1.0) / ((d + 1) * (M[torch.arange(M.shape[0]), jdx] - 1.0))  # [B_active,]
        new_u_active = (1 - step_size.unsqueeze(1)) * u[active_mask]  # [B_active, N]
        new_u_active.scatter_add_(1, jdx.unsqueeze(1), step_size.unsqueeze(1)) 
        err_active = torch.linalg.norm(new_u_active - u[active_mask], dim=1)  # [B_active,]
            
        u[active_mask] = new_u_active
        err[active_mask] = err_active
        active_mask = err > tol

    c = torch.einsum('bn,bnd->bd', u, clusters)  # [B, 3]
    intermediate = torch.einsum('bnd,bn,bnm->bdm', clusters, u, clusters)  # [B, 3, 3]
    A = torch.linalg.inv(intermediate - torch.einsum('bd,bm->bdm', c, c)) / d  # [B, 3, 3]

    if rgbs is not None:
        print("\n Ellipsoids generated \n")
        rgbs_np = rgbs.cpu().numpy()
        rgbs_np = rgbs_np[valid_neighbors]
        return A, c, torch.from_numpy(rgbs_np)
    else:
        return A, c, None
