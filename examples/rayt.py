import argparse
import math
import os
import time
from typing import Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser

from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization, _rasterization
from gsplat.cuda._torch_impl import (
    _quat_scale_to_covar_preci, 
    _fully_fused_projection2, 
    gaussian_to_ellipse, 
    cull_mask,
)

def generate_rays(c2w: torch.Tensor, K: torch.Tensor, width: int, height: int, device):
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



def signed_distance(current_position: torch.Tensor, means: torch.Tensor, r_a: torch.Tensor, r_b: torch.Tensor, axes_a: torch.Tensor, axes_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    diff = current_position.unsqueeze(1) - means

    
    axes_a_normalized = axes_a 
    axes_b_normalized = axes_b   

    projection_a = torch.einsum('pni,ni->pn', diff, axes_a_normalized)  
    projection_b = torch.einsum('pni,ni->pn', diff, axes_b_normalized)

    projected_point = (
        projection_a.unsqueeze(-1) * axes_a_normalized.unsqueeze(0) +  
        projection_b.unsqueeze(-1) * axes_b_normalized.unsqueeze(0)   
    ) 

    scaled_a = projection_a / r_a  
    scaled_b = projection_b / r_b  

    scaling_factor = torch.sqrt(scaled_a**2 + scaled_b**2)
    #scaling_factor = torch.where(scaling_factor < 1e-8, torch.tensor(1e-8, device=scaling_factor.device), scaling_factor)

    closest_a = (projection_a / scaling_factor)
    closest_b = (projection_b / scaling_factor)
    closest_point_boundary = (
        closest_a.unsqueeze(-1) * axes_a_normalized.unsqueeze(0) +  
        closest_b.unsqueeze(-1) * axes_b_normalized.unsqueeze(0)   
    )
    
    distance_projected = torch.norm(diff - projected_point, dim=-1)

    distance_boundary = torch.norm(diff - closest_point_boundary, dim=-1)

    inside_mask = scaled_a**2 + scaled_b**2 <= 1

    distances = torch.where(inside_mask, distance_projected, distance_boundary)

    sdf_min, sdf_indices = torch.min(distances, dim=1)

    return sdf_min, sdf_indices
    

def ray_march(
    ro: torch.Tensor,  # Ray origins [H, W, 3]
    rd: torch.Tensor,  # Ray directions [H, W, 3]
    r_a: torch.Tensor,  # Gaussian centers [N, 3]
    r_b: torch.Tensor,  # Gaussian quaternions [N, 4]
    axes_a: torch.Tensor,  # Gaussian scales [N, 3]
    axes_b: torch.Tensor,  # Gaussian scales [N, 3]
    means: torch.Tensor,
    opacities: torch.Tensor,  # Gaussian opacities [N]
    colors: torch.Tensor,  # Gaussian colors [N, 1, 3]
    max_steps: int = 128,
    min_hit_distance: float = 0.001,
    max_trace_distance: float = 5.0,
) -> torch.Tensor:
    """
    Optimized ray marching with signed distance function for Gaussian blending.
    """
    H, W = ro.shape[:2]  # Image dimensions
    N = means.shape[0]  # Number of Gaussians
    # Initialize outputs and active rays
    total_distance = torch.zeros((H, W), device=ro.device)  # Distance along each ray
    hit_color = torch.zeros((H, W, 3), device=ro.device)  # Final color output
    active_mask = torch.ones((H, W), dtype=torch.bool, device=ro.device)  # Active rays mask

    for step in range(max_steps):
        if not active_mask.any():
            break  # Exit if no active rays remain

        current_position = ro + total_distance[..., None] * rd  # [H, W, 3]

        active_indices = torch.nonzero(active_mask, as_tuple=True)  # Active ray indices
        active_positions = current_position[active_indices]  # [num_active, 3]  

        if active_positions.numel() == 0:  # No active rays left
            break

        closest_distances, closest_gaussians = signed_distance(active_positions, means, r_a, r_b, axes_a, axes_b)  # [num_active, N]

        hit_mask = closest_distances < min_hit_distance

        if hit_mask.any():
            hit_active_indices = active_indices[0][hit_mask], active_indices[1][hit_mask]
            hit_gaussian_indices = closest_gaussians[hit_mask]

            hit_colors = colors[hit_gaussian_indices, 0]  # [num_hits, 3]
            hit_color[hit_active_indices] = hit_colors 

        total_distance[active_indices] += closest_distances
        active_mask[active_indices] &= total_distance[active_indices] < max_trace_distance

    return hit_color, total_distance


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    if args.ckpt == None:
        means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []

        means = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]], device=device, dtype=torch.float32)  # Gaussian centers
        quats = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], device=device, dtype=torch.float32)  # Identity quaternions
        scales = torch.tensor([[0.0001, 0.5, 0.5], [0.5, 0.5, 0.0001], [0.5, 0.0001, 0.5]], device=device, dtype=torch.float32)  # Gaussian scales
        opacities = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32)  # Fully opaque
        colors = torch.tensor([[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]]], device=device, dtype=torch.float32)  # Red and Green with extra dimension
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
        print("Number of Gaussians:", len(means))
    else:
        means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
        for ckpt_path in args.ckpt:
            ckpt = torch.load(ckpt_path, map_location=device)["splats"]
            means.append(ckpt["means"])
            quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
            scales.append(torch.exp(ckpt["scales"]))
            opacities.append(torch.sigmoid(ckpt["opacities"]))
            sh0.append(ckpt["sh0"])
            shN.append(ckpt["shN"])
        means = torch.cat(means, dim=0)
        quats = torch.cat(quats, dim=0)
        scales = torch.cat(scales, dim=0)
        opacities = torch.cat(opacities, dim=0)
        sh0 = torch.cat(sh0, dim=0)
        shN = torch.cat(shN, dim=0)
        colors = torch.cat([sh0, shN], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
        print("Number of Gaussians:", len(means))

    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        if args.backend == "gsplat":
            rasterization_fn = rasterization
        elif args.backend == "inria":
            from gsplat import rasterization_inria_wrapper
            rasterization_fn = rasterization_inria_wrapper
            
        elif args.backend == "raymarch":

            r_a, r_b, axes_a, axes_b = gaussian_to_ellipse(means, quats, scales)
            

            ro, rd = generate_rays(c2w, K, width, height, device)


            ro_m = ro[0,0]
            rd_m = rd[0,0]
            
            mask = cull_mask(ro_m, rd_m, r_a, r_b, means)
            print(mask)


            render_colors, total_distance = ray_march(ro, rd, r_a, r_b, axes_a, axes_b, means, opacities, colors)
            
            K_inv = torch.linalg.inv(K)  # Compute inverse of K

            i, j = torch.meshgrid(torch.arange(width, device=K.device),
                                torch.arange(height, device=K.device),
                                indexing="ij")

            pixel_coords = torch.stack([i, j, torch.ones_like(i)], dim=-1).float()  # [H, W, 3]
            normalized_coords = (K_inv @ pixel_coords.reshape(-1, 3).T).T.reshape(height, width, 3)  # [H, W, 3]
            denominator = torch.sqrt(normalized_coords[..., 0]**2 + normalized_coords[..., 1]**2 + 1)  # [H, W]
            depth = total_distance / denominator  # [H, W]

            min_depth = 0.0
            max_depth = depth.max() 
            depth_norm = (depth - min_depth) / (max_depth - min_depth)
            depth_expanded = 1.0 - depth_norm.unsqueeze(-1).expand(-1, -1, 3)


            return depth_expanded.cpu().numpy()

        else:
            raise ValueError

        render_colors, render_alphas, meta = rasterization_fn(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=sh_degree,
            render_mode="RGB",
            # this is to speedup large-scale rendering by skipping far-away Gaussians.
            # radius_clip=3,
        )
        render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
        return render_rgbs


    width, height = 800, 800
    # camera_state = nerfview.CameraState(
    #     fov=1.3089969389957472,
    #     aspect=1.6985294117647058,
    #     c2w=np.array([
    #         [ 1.38156355e-01, -2.42040851e-03, -9.90407473e-01,  2.78085955e+00],
    #         [ 9.90410431e-01,  3.37632568e-04,  1.38155942e-01, -3.87913340e-01],
    #         [-3.33066907e-16, -9.99997014e-01,  2.44384392e-03, -6.86180879e-03],
    #         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
    #     ])
    # )
    camera_state = nerfview.CameraState(
        fov=1.3089969389957472,
        aspect=1.6985294117647058,
        c2w=np.array([
            [-8.31469745e-01,  3.80353201e-01, -4.04956177e-01,  3.68168529e-01],
            [ 5.55570034e-01,  5.69239088e-01, -6.06060061e-01,  4.06304876e-01],
            [-1.38777878e-16, -7.28902122e-01, -6.84617920e-01,  5.59059906e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
        ])
    )
    rendered_image = viewer_render_fn(camera_state, (width, height))

    # Save the rendered image
    os.makedirs(args.output_dir, exist_ok=True)

    if args.backend == "gsplat":
        imageio.imsave(
            f"{args.output_dir}/render_rasterization.png",
            (rendered_image * 255).astype(np.uint8),
        )
    elif args.backend == "raymarch":
        imageio.imsave(
            f"{args.output_dir}/render_raymarch.png",
            (rendered_image * 255).astype(np.uint8),
        )

    server = viser.ViserServer(port=8080, verbose=False)
    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)

if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=0 python simple_viewer.py \
        --ckpt results/garden/ckpts/ckpt_3499_rank0.pt results/garden/ckpts/ckpt_3499_rank1.pt 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument("--backend", type=str, default="gsplat", help="gsplat, inria, raymarch")
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)