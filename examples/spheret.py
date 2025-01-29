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
    generate_rays,
    gaussian_to_ellipse,
    cull_mask,
    signed_distance,
    sphere_trace,
    calculate_depth,
    generate_depth_image
)


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
            
        elif args.backend == "sphere_trace":            
            origins, directions = generate_rays(c2w, K, width, height)

            hit_color, total_distance = sphere_trace(
                means,
                quats,
                scales,
                colors,
                origins,
                directions
            )

            torch.save(total_distance, "results/distance.pt")
            torch.save(hit_color, "result/colors.pt")

            depth = calculate_depth(K, total_distance, width, height)
            depth_img = generate_depth_image(depth, total_distance, (9.5))

            return depth_img.cpu().numpy()

        elif args.backend == "load":
            total_distance = torch.load("sdf.pt")
            render_color = torch.load("colors.pt")
            
            depth = calculate_depth(K, total_distance, width, height)
            depth_img = generate_depth_image(depth, total_distance, (9.5))

            return depth_img.cpu().numpy()

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
    #         aspect=1.6985294117647058,
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
            f"{args.output_dir}/render_sphere_trace.png",
            (rendered_image * 255).astype(np.uint8),
        )
    elif args.backend == "load":
        imageio.imsave(
            f"{args.output_dir}/render_sphere_trace_load.png",
            (rendered_image * 255).astype(np.uint8),
        )

    # server = viser.ViserServer(port=8080, verbose=False)
    # _ = nerfview.Viewer(
    #     server=server,
    #     render_fn=viewer_render_fn,
    #     mode="rendering",
    # )
    # print("Viewer running... Ctrl+C to exit.")
    # time.sleep(100000)

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
    parser.add_argument("--backend", type=str, default="gsplat", help="gsplat, inria, sphere_trace")
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)